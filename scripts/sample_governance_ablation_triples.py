#!/usr/bin/env python3
"""Sample triples for governed-vs-flat manual governance ablation.

Groups:
- A governed_only: admitted by governed KG, absent from flat KG
- B flat_only: admitted by flat KG, absent from governed KG
- C shared: present in both
- D revised: governance decisions with action == revise

The output is intentionally annotation-friendly: each row includes the
triple, source document, available evidence/original surfaces, and blank
fields for human or LLM-assisted validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RELATION_ALIASES = {
    "USED_FOR": "Used-for",
    "USES": "Used-for",
    "UTILIZES": "Used-for",
    "APPLIED_TO": "Used-for",
    "APPLIES_TO": "Used-for",
    "FEATURE_OF": "Feature-of",
    "HAS_FEATURE": "Feature-of",
    "PROPERTY_OF": "Feature-of",
    "PART_OF": "Part-of",
    "IS_PART_OF": "Part-of",
    "COMPONENT_OF": "Part-of",
    "HYPONYM_OF": "Hyponym-of",
    "IS_A": "Hyponym-of",
    "TYPE_OF": "Hyponym-of",
    "SUBTYPE_OF": "Hyponym-of",
    "COMPARE": "Compare",
    "COMPARES": "Compare",
    "COMPARED_TO": "Compare",
    "CONJUNCTION": "Conjunction",
    "AND": "Conjunction",
    "COMBINED_WITH": "Conjunction",
    "EVALUATE_FOR": "Evaluate-for",
    "EVALUATED_FOR": "Evaluate-for",
    "EVALUATES": "Evaluate-for",
}


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _kg(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("knowledge_graph", data)


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _canonical_relation(relation: Any) -> str:
    raw = str(relation or "").strip()
    norm = raw.upper().replace("-", "_").replace(" ", "_")
    return RELATION_ALIASES.get(norm, raw)


def _triple_key(triple: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        _norm_text(triple.get("subject")),
        _canonical_relation(triple.get("relation")),
        _norm_text(triple.get("object")),
    )


def _entity_lookup(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {entity.get("id"): entity for entity in _kg(data).get("entities", [])}


def _surface(entity_id: str, entities: Dict[str, Dict[str, Any]]) -> str:
    entity = entities.get(entity_id) or {}
    labels = entity.get("labels") or []
    return labels[0] if labels else entity_id


def _source_context_lookup(scierc_path: str) -> Dict[str, str]:
    path = Path(scierc_path)
    if not path.exists():
        return {}
    contexts: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        doc_key = record.get("doc_key")
        sentences = record.get("sentences") or []
        if doc_key:
            contexts[doc_key] = " ".join(
                " ".join(str(token) for token in sentence)
                for sentence in sentences
            )
    return contexts


def _expand_evidence(evidence: str, source_context: str) -> str:
    evidence = str(evidence or "").strip()
    source_context = str(source_context or "").strip()
    if not evidence:
        return source_context
    if "..." not in evidence:
        return evidence
    if not source_context:
        return evidence.replace("...", "").strip()

    parts = [part.strip() for part in evidence.split("...") if part.strip()]
    if not parts:
        return source_context

    context_lower = source_context.lower()
    positions = []
    for part in parts:
        pos = context_lower.find(part.lower())
        if pos == -1:
            return source_context
        positions.append((pos, pos + len(part)))
    start = min(pos[0] for pos in positions)
    end = max(pos[1] for pos in positions)
    return source_context[start:end].strip()


def _row(
    *,
    group: str,
    triple: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
    counterpart: Optional[Dict[str, Any]] = None,
    audit: Optional[Dict[str, Any]] = None,
    source_contexts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    metadata = triple.get("metadata") if isinstance(triple.get("metadata"), dict) else {}
    source_document = triple.get("source") or metadata.get("source_document")
    source_context = (source_contexts or {}).get(source_document, "")
    raw_evidence = metadata.get("evidence") or triple.get("evidence") or ""
    row = {
        "group": group,
        "subject": triple.get("subject"),
        "relation": triple.get("relation"),
        "object": triple.get("object"),
        "subject_label": metadata.get("original_subject") or _surface(triple.get("subject"), entities),
        "object_label": metadata.get("original_object") or _surface(triple.get("object"), entities),
        "canonical_relation": _canonical_relation(triple.get("relation")),
        "source_document": source_document,
        "confidence": triple.get("confidence"),
        "evidence": _expand_evidence(raw_evidence, source_context),
        "source_context": source_context,
        "counterpart_subject": counterpart.get("subject") if counterpart else "",
        "counterpart_relation": counterpart.get("relation") if counterpart else "",
        "counterpart_object": counterpart.get("object") if counterpart else "",
        "governance_action": audit.get("action") if audit else "",
        "governance_rationale": audit.get("rationale") if audit else "",
        "original_subject_before_revision": "",
        "original_relation_before_revision": "",
        "original_object_before_revision": "",
        "annotation_supported_by_source": "",
        "annotation_correct_relation": "",
        "annotation_correct_endpoints": "",
        "annotation_useful_for_graph_or_qa": "",
        "annotation_decision_quality": "",
        "annotation_notes": "",
    }
    if audit and audit.get("action") == "revise":
        original = audit.get("triple") or {}
        row.update(
            {
                "original_subject_before_revision": original.get("subject", ""),
                "original_relation_before_revision": original.get("relation", ""),
                "original_object_before_revision": original.get("object", ""),
            }
        )
    return row


def _sample(items: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n <= 0 or len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--governed", default="evaluation/results/governed_created_50docs_clean.json")
    parser.add_argument("--flat", default="evaluation/results/ungoverned_created_50docs_clean.json")
    parser.add_argument("--output-json", default="evaluation/results/governance_ablation_samples_50docs.json")
    parser.add_argument("--output-csv", default="evaluation/results/governance_ablation_samples_50docs.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-governed-only", type=int, default=100)
    parser.add_argument("--n-flat-only", type=int, default=100)
    parser.add_argument("--n-shared", type=int, default=50)
    parser.add_argument("--scierc-path", default="evaluation/datasets/scierc/test.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    governed = _load(args.governed)
    flat = _load(args.flat)
    governed_entities = _entity_lookup(governed)
    flat_entities = _entity_lookup(flat)
    source_contexts = _source_context_lookup(args.scierc_path)

    governed_triples = _kg(governed).get("triples", [])
    flat_triples = _kg(flat).get("triples", [])
    governed_by_key = {_triple_key(triple): triple for triple in governed_triples}
    flat_by_key = {_triple_key(triple): triple for triple in flat_triples}

    governed_keys = set(governed_by_key)
    flat_keys = set(flat_by_key)
    governed_only = sorted(governed_keys - flat_keys)
    flat_only = sorted(flat_keys - governed_keys)
    shared = sorted(governed_keys & flat_keys)

    audit_log = governed.get("audit_log", [])
    revised = [
        entry
        for entry in audit_log
        if isinstance(entry, dict)
        and entry.get("action") == "revise"
        and isinstance(entry.get("revised_triple"), dict)
    ]

    rows: List[Dict[str, Any]] = []
    for key in _sample([{"key": key} for key in governed_only], args.n_governed_only, rng):
        triple = governed_by_key[key["key"]]
        rows.append(_row(group="A_governed_only", triple=triple, entities=governed_entities, source_contexts=source_contexts))
    for key in _sample([{"key": key} for key in flat_only], args.n_flat_only, rng):
        triple = flat_by_key[key["key"]]
        rows.append(_row(group="B_flat_only", triple=triple, entities=flat_entities, source_contexts=source_contexts))
    for key in _sample([{"key": key} for key in shared], args.n_shared, rng):
        triple = governed_by_key[key["key"]]
        rows.append(
            _row(
                group="C_shared",
                triple=triple,
                entities=governed_entities,
                counterpart=flat_by_key[key["key"]],
                source_contexts=source_contexts,
            )
        )
    for entry in revised:
        rows.append(
            _row(
                group="D_revised",
                triple=entry["revised_triple"],
                entities=governed_entities,
                audit=entry,
                source_contexts=source_contexts,
            )
        )

    summary = {
        "governed_path": args.governed,
        "flat_path": args.flat,
        "seed": args.seed,
        "scierc_path": args.scierc_path,
        "population": {
            "governed_triples": len(governed_triples),
            "flat_triples": len(flat_triples),
            "governed_only": len(governed_only),
            "flat_only": len(flat_only),
            "shared": len(shared),
            "revised": len(revised),
        },
        "sample_counts": {
            "A_governed_only": sum(row["group"] == "A_governed_only" for row in rows),
            "B_flat_only": sum(row["group"] == "B_flat_only" for row in rows),
            "C_shared": sum(row["group"] == "C_shared" for row in rows),
            "D_revised": sum(row["group"] == "D_revised" for row in rows),
        },
        "annotation_rubric": {
            "annotation_supported_by_source": "yes/no/partial",
            "annotation_correct_relation": "yes/no/partial",
            "annotation_correct_endpoints": "yes/no/partial",
            "annotation_useful_for_graph_or_qa": "yes/no",
            "annotation_decision_quality": "good/bad/unclear; for flat-only, whether governance rejection was correct",
        },
        "rows": rows,
    }

    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    _write_csv(Path(args.output_csv), rows)
    print(json.dumps({k: summary[k] for k in ("population", "sample_counts")}, indent=2))


if __name__ == "__main__":
    main()
