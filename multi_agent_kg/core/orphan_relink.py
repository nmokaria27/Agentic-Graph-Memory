"""
Orphan relink pass: targeted recall for entities left without connections.

The extraction funnel is asymmetric by design — the entity extractor is
recall-oriented while the relation extractor is precision-oriented (its
prompts skip any binding without an explicit linguistic marker). The result
is a tail of catalog entities with zero edges. This pass revisits each
orphan with its own source text and a small set of candidate partner
entities, and asks the LLM for relations involving that orphan only.

All recovered triples enter through `governed_kg.propose_triple()` so
governance routing, confidence policies, and the audit log apply unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import re

from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import Entity
from multi_agent_kg.llm.openai_client import chat_completion_json

# Value-like entity types that, if still orphaned after relinking, are folded
# into the metadata of a host entity instead of remaining free-standing nodes.
VALUE_LIKE_TYPES = {
    "DATE",
    "FINANCIAL_METRIC",
    "MONETARY_VALUE",
    "QUANTITY",
    "PERCENTAGE",
    "NUMERIC_VALUE",
    "MEASUREMENT",
}

ORPHAN_RELINK_PROMPT = """You are recovering missed relations for specific entities.

SOURCE TEXT:
{source_text}

DISCONNECTED ENTITIES (find relations involving these):
{orphan_catalog}

CANDIDATE PARTNER ENTITIES (already in the graph; prefer these as the other endpoint):
{candidate_catalog}

RULES:
1. Only extract relations that the SOURCE TEXT directly supports. Do NOT use
   background knowledge, plausibility, or co-occurrence alone.
2. Each triple must involve at least one DISCONNECTED entity.
3. Copy "subject_id" and "object_id" EXACTLY from the "id" fields above.
4. Subject and object must be DIFFERENT entities.
5. Relation names: concise UPPER_SNAKE_CASE predicates.
6. If the text supports nothing for an entity, return nothing for it.

Return JSON:
{{
    "triples": [
        {{
            "subject_id": "<id>",
            "relation": "<RELATION>",
            "object_id": "<id>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text snippet>"
        }}
    ]
}}

Return ONLY the JSON."""


def _entity_source_segments(entity: Entity) -> Set[str]:
    metadata = entity.metadata or {}
    segments = set(metadata.get("source_segments") or [])
    if metadata.get("source_segment"):
        segments.add(metadata["source_segment"])
    return segments


def _entity_source_text(entity: Entity) -> str:
    metadata = entity.metadata or {}
    texts = metadata.get("source_texts") or []
    if not texts and metadata.get("source_text"):
        texts = [metadata["source_text"]]
    return "\n".join(str(t) for t in texts if t)


def _catalog(entities: List[Entity]) -> str:
    import json

    return json.dumps(
        [
            {
                "id": e.id,
                "text": (e.labels[0] if e.labels else e.id.replace("_", " ")),
                "type": e.type or "",
            }
            for e in entities
        ],
        indent=2,
    )


def _normalize_surface(text: str) -> str:
    return " ".join(
        str(text)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _degenerate_pair(subject: str, obj: str, relation: str) -> bool:
    subj = _normalize_surface(subject)
    obj_norm = _normalize_surface(obj)
    if not subj or not obj_norm:
        return True
    if subj == obj_norm:
        return True
    subj_tokens = subj.split()
    obj_tokens = obj_norm.split()
    shorter, longer = (subj, obj_norm) if len(subj_tokens) <= len(obj_tokens) else (obj_norm, subj)
    if len(shorter.split()) < 2:
        return False
    relation_norm = relation.strip().upper().replace("-", "_").replace(" ", "_")
    if relation_norm in {"HYPONYM_OF", "SUBTYPE_OF", "TYPE_OF", "INSTANCE_OF", "PART_OF", "IS_PART_OF", "COMPONENT_OF"}:
        return False
    return re.search(rf"(?<!\w){re.escape(shorter)}(?!\w)", longer) is not None


def _degenerate_triple(triple: Dict[str, Any], kg_entities: Dict[str, Entity]) -> bool:
    subject = str(triple.get("subject_id") or "").strip()
    obj = str(triple.get("object_id") or "").strip()
    if subject == obj:
        return True
    subject_entity = kg_entities.get(subject)
    object_entity = kg_entities.get(obj)
    subject_text = subject_entity.labels[0] if subject_entity and subject_entity.labels else subject
    object_text = object_entity.labels[0] if object_entity and object_entity.labels else obj
    return _degenerate_pair(subject_text, object_text, str(triple.get("relation") or ""))


def _candidate_partners(
    orphan: Entity,
    governed_kg: GovernedKnowledgeGraph,
    vector_store: Optional[Any],
    limit: int = 8,
) -> List[Entity]:
    """Connected entities likely related to the orphan.

    Union of (a) entities sharing a source segment with the orphan and
    (b) top entity-index hits for the orphan's text, restricted to entities
    that already have connections (so new edges attach to the graph core).
    """
    kg = governed_kg.kg
    connected: Set[str] = set()
    for triple in kg.triples:
        connected.add(triple.subject)
        connected.add(triple.object)

    orphan_segments = _entity_source_segments(orphan)
    candidates: List[Entity] = []
    seen: Set[str] = set()

    for entity_id in connected:
        entity = kg.entities.get(entity_id)
        if entity is None or entity_id == orphan.id:
            continue
        if orphan_segments & _entity_source_segments(entity):
            candidates.append(entity)
            seen.add(entity_id)
            if len(candidates) >= limit:
                break

    if vector_store is not None and len(candidates) < limit:
        query = " ".join([orphan.id.replace("_", " ")] + list(orphan.labels))
        try:
            hits = vector_store.entity_index.search(query, top_k=limit * 2)
        except Exception:
            hits = []
        for entity_id, _score in hits:
            if entity_id in seen or entity_id == orphan.id or entity_id not in connected:
                continue
            entity = kg.entities.get(entity_id)
            if entity is not None:
                candidates.append(entity)
                seen.add(entity_id)
            if len(candidates) >= limit:
                break

    return candidates


def relink_orphans(
    governed_kg: GovernedKnowledgeGraph,
    llm_config: Optional[LLMConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
    vector_store: Optional[Any] = None,
    batch_size: int = 8,
    max_orphans: Optional[int] = None,
) -> Dict[str, Any]:
    """One targeted relink round over all orphan entities.

    Orphans are grouped by their primary source segment so each LLM call
    shares one source text. Returns a stats dict for the funnel summary.
    """
    llm_config = llm_config or LLMConfig()
    retrieval_config = retrieval_config or RetrievalConfig()
    kg = governed_kg.kg

    orphans = kg.get_orphan_entities()
    if max_orphans is not None:
        orphans = orphans[:max_orphans]
    relinkable_orphans = [
        orphan for orphan in orphans
        if (orphan.type or "").upper() not in VALUE_LIKE_TYPES
    ]
    stats: Dict[str, Any] = {
        "orphans_before": len(orphans),
        "llm_calls": 0,
        "triples_proposed": 0,
        "triples_committed": 0,
        "orphans_skipped_no_text": 0,
        "orphans_skipped_value_like": len(orphans) - len(relinkable_orphans),
    }
    if not relinkable_orphans:
        stats["orphans_after"] = len(kg.get_orphan_entities())
        return stats

    # Group orphans by primary source segment (shared context per LLM call).
    groups: Dict[str, List[Entity]] = {}
    for orphan in relinkable_orphans:
        segments = sorted(_entity_source_segments(orphan))
        key = segments[0] if segments else "__no_segment__"
        groups.setdefault(key, []).append(orphan)

    for _segment_id, group in groups.items():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            source_text = "\n\n".join(
                dict.fromkeys(  # dedupe identical snippets, keep order
                    text for text in (_entity_source_text(e) for e in batch) if text
                )
            )
            if not source_text.strip():
                stats["orphans_skipped_no_text"] += len(batch)
                continue

            partners: List[Entity] = []
            partner_ids: Set[str] = set()
            for orphan in batch:
                for partner in _candidate_partners(orphan, governed_kg, vector_store):
                    if partner.id not in partner_ids:
                        partners.append(partner)
                        partner_ids.add(partner.id)
            # Orphans in the same batch can also relate to each other.
            allowed_ids = {e.id for e in batch} | partner_ids

            prompt = ORPHAN_RELINK_PROMPT.format(
                source_text=source_text[:6000],
                orphan_catalog=_catalog(batch),
                candidate_catalog=_catalog(partners[:24]),
            )
            try:
                result = chat_completion_json(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You recover missed knowledge-graph relations. "
                                "Extract only text-supported triples and return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=llm_config.model,
                    temperature=0.1,
                )
            except Exception as exc:
                print(f"  WARNING: orphan relink LLM call failed: {exc}")
                continue
            stats["llm_calls"] += 1

            triples = result.get("triples", []) if isinstance(result, dict) else []
            batch_ids = {e.id for e in batch}
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                subject = str(triple.get("subject_id") or "").strip()
                obj = str(triple.get("object_id") or "").strip()
                relation = str(triple.get("relation") or "").strip()
                if not subject or not obj or not relation or subject == obj:
                    continue
                if subject not in allowed_ids or obj not in allowed_ids:
                    continue
                if subject not in batch_ids and obj not in batch_ids:
                    continue
                if _degenerate_triple(triple, kg.entities):
                    continue
                stats["triples_proposed"] += 1
                decision = governed_kg.propose_triple(
                    subject=subject,
                    relation=relation,
                    obj=obj,
                    confidence=float(triple.get("confidence", 0.6) or 0.6),
                    source="orphan_relink",
                    metadata={
                        "evidence": str(triple.get("evidence") or ""),
                        "orphan_relink": True,
                    },
                )
                if decision.committed:
                    stats["triples_committed"] += 1

    stats["orphans_after"] = len(kg.get_orphan_entities())
    return stats


def fold_value_orphans(
    governed_kg: GovernedKnowledgeGraph,
    value_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Fold remaining value-like orphans into a host entity's metadata.

    The host is a connected entity sharing a source segment with the orphan.
    The orphan node is removed; its surface form is preserved under the
    host's `attributes` metadata. Orphans with no host are left untouched.
    """
    value_types = value_types or VALUE_LIKE_TYPES
    kg = governed_kg.kg
    stats: Dict[str, Any] = {"folded": 0, "no_host": 0, "candidates": 0}

    connected: Set[str] = set()
    for triple in kg.triples:
        connected.add(triple.subject)
        connected.add(triple.object)

    # Pre-index connected entities by source segment.
    segment_hosts: Dict[str, str] = {}
    for entity_id in connected:
        entity = kg.entities.get(entity_id)
        if entity is None:
            continue
        for segment in _entity_source_segments(entity):
            segment_hosts.setdefault(segment, entity_id)

    for orphan in kg.get_orphan_entities():
        if (orphan.type or "").upper() not in value_types:
            continue
        stats["candidates"] += 1
        host_id = None
        for segment in _entity_source_segments(orphan):
            if segment in segment_hosts:
                host_id = segment_hosts[segment]
                break
        if host_id is None:
            stats["no_host"] += 1
            continue
        host = kg.entities[host_id]
        attributes = host.metadata.setdefault("attributes", {})
        attributes[orphan.id] = {
            "value": orphan.labels[0] if orphan.labels else orphan.id.replace("_", " "),
            "type": orphan.type,
            "folded_from_orphan": True,
        }
        del kg.entities[orphan.id]
        for domain in governed_kg.org_chart.domains:
            domain.entity_ids.discard(orphan.id)
        stats["folded"] += 1

    return stats


__all__ = ["relink_orphans", "fold_value_orphans", "VALUE_LIKE_TYPES"]
