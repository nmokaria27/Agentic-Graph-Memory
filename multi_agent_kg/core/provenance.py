"""Unified provenance for admitted entities and triples.

Every admitted entity and triple should trace back to its exact source: the
document, the chunk/segment it came from, the character span within that
document, the agent that produced it, and how its confidence was derived.

This module defines one metadata schema (``metadata["provenance"]``) and pure
helpers to build, merge, and read it. It deliberately holds no LLM calls and no
KG references so it stays unit-testable in isolation.

Schema (stored under ``metadata[PROVENANCE_KEY]``)::

    {
        "refs": [                         # one per source location (>=1)
            {
                "document_id": str,
                "segment_id":  str | None,
                "chunk_index": int | None,
                "char_start":  int | None,   # offset into the document
                "char_end":    int | None,
                "source_path": str | None,
                "snippet":     str | None,   # short supporting text
            },
            ...
        ],
        "extractor":         str,         # agent that produced the item
        "confidence":        float,
        "confidence_source": str,         # how confidence was derived
        "evidence_sentences": [str, ...], # triples only; [] otherwise
    }

An entity that survives coreference/dedup accumulates one ref per mention; a
triple accumulates one ref per supporting span. ``merge`` unions refs and keeps
the higher confidence, so provenance is preserved across merges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

PROVENANCE_KEY = "provenance"

# How a confidence value was arrived at, for audit.
CONFIDENCE_EXTRACTION = "extraction"      # raw score from the extracting agent
CONFIDENCE_VERIFICATION = "verification"  # adjusted by the verification gate
CONFIDENCE_REFINEMENT = "refinement"      # adjusted during refinement loop
CONFIDENCE_RELINK = "orphan_relink"       # recovered by the orphan relink pass


def source_ref(
    document_id: str,
    *,
    segment_id: Optional[str] = None,
    chunk_index: Optional[int] = None,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    source_path: Optional[str] = None,
    snippet: Optional[str] = None,
    document_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single source-location reference.

    ``document_date`` is the real-world date the source content is from/about
    (e.g. a dated session transcript, a versioned report) — not ingestion
    wall-clock time. None when the source has no natural date. Lets
    freshness-sensitive conflict resolution judge recency (GB-2).
    """
    return {
        "document_id": document_id,
        "segment_id": segment_id,
        "chunk_index": chunk_index,
        "char_start": char_start,
        "char_end": char_end,
        "source_path": source_path,
        "snippet": (snippet[:280] if snippet else None),
        "document_date": document_date,
    }


def build_provenance(
    *,
    refs: List[Dict[str, Any]],
    extractor: str,
    confidence: float,
    confidence_source: str = CONFIDENCE_EXTRACTION,
    evidence_sentences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble a provenance record from one or more source refs."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "refs": [r for r in (refs or []) if r],
        "extractor": extractor,
        "confidence": conf,
        "confidence_source": confidence_source,
        "evidence_sentences": list(evidence_sentences or []),
    }


def _ref_key(ref: Dict[str, Any]) -> tuple:
    """Identity of a ref for de-duplication when merging."""
    return (
        ref.get("document_id"),
        ref.get("segment_id"),
        ref.get("char_start"),
        ref.get("char_end"),
    )


def merge(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Union two provenance records (e.g. across a coref/dedup merge).

    Refs are unioned (deduplicated by document/segment/span), confidence is the
    max of the two, and its source is taken from whichever side supplied that
    max. Evidence sentences are unioned preserving order.
    """
    a = a or {"refs": [], "extractor": "", "confidence": 0.0,
              "confidence_source": CONFIDENCE_EXTRACTION, "evidence_sentences": []}
    b = b or {"refs": [], "extractor": "", "confidence": 0.0,
              "confidence_source": CONFIDENCE_EXTRACTION, "evidence_sentences": []}

    seen = set()
    refs: List[Dict[str, Any]] = []
    for ref in list(a.get("refs", [])) + list(b.get("refs", [])):
        k = _ref_key(ref)
        if k in seen:
            continue
        seen.add(k)
        refs.append(ref)

    if float(b.get("confidence", 0.0)) > float(a.get("confidence", 0.0)):
        confidence = float(b.get("confidence", 0.0))
        confidence_source = b.get("confidence_source", CONFIDENCE_EXTRACTION)
    else:
        confidence = float(a.get("confidence", 0.0))
        confidence_source = a.get("confidence_source", CONFIDENCE_EXTRACTION)

    evidence: List[str] = []
    for s in list(a.get("evidence_sentences", [])) + list(b.get("evidence_sentences", [])):
        if s and s not in evidence:
            evidence.append(s)

    return {
        "refs": refs,
        "extractor": a.get("extractor") or b.get("extractor") or "",
        "confidence": confidence,
        "confidence_source": confidence_source,
        "evidence_sentences": evidence,
    }


def attach(metadata: Optional[Dict[str, Any]], provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Attach/merge provenance into a metadata dict, returning the dict.

    If the metadata already carries provenance, the two are merged so no source
    location is lost. Mutates and returns ``metadata`` (creating it if None).
    """
    metadata = metadata if metadata is not None else {}
    existing = metadata.get(PROVENANCE_KEY)
    metadata[PROVENANCE_KEY] = merge(existing, provenance) if existing else provenance
    return metadata


def get(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Read provenance from a metadata dict, or None."""
    if not metadata:
        return None
    return metadata.get(PROVENANCE_KEY)


def describe(metadata: Optional[Dict[str, Any]]) -> str:
    """One-line human-readable provenance summary for audit/logging."""
    prov = get(metadata)
    if not prov:
        return "<no provenance>"
    refs = prov.get("refs", [])
    locs = ", ".join(
        f"{r.get('document_id')}[{r.get('char_start')}:{r.get('char_end')}]"
        for r in refs[:3]
    )
    more = "" if len(refs) <= 3 else f" (+{len(refs) - 3} more)"
    return (
        f"{prov.get('extractor', '?')} conf={prov.get('confidence', 0):.2f}"
        f"/{prov.get('confidence_source', '?')} <- {locs}{more}"
    )
