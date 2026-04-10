"""
Knowledge Graph Operations: Merge, Diff, and Incremental Update.

Provides the data-layer primitives for:
- Comparing two KGs (diff)
- Merging a delta KG into a base KG with conflict detection
- Serializing / deserializing KG snapshots
- Entity resolution across KG boundaries (fuzzy + embedding-ready)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import Entity, KnowledgeGraph, Triple


# ---------------------------------------------------------------------------
# Fuzzy entity matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


_GREEK_TO_ASCII = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ω": "omega", "τ": "tau", "σ": "sigma",
}


def normalize_entity_name(name: str) -> str:
    """Canonical entity name normalization for matching.

    Converts hyphens, underscores, and extra spaces to a single space,
    lowercases, strips, and normalizes Greek letters (α→alpha, etc.).
    This ensures that ``IL-6``, ``il_6``, ``il 6`` all map to ``il 6``
    and ``TNF-α`` matches ``tnf_alpha``.
    """
    name = name.lower().strip()
    for greek, ascii_ in _GREEK_TO_ASCII.items():
        name = name.replace(greek, " " + ascii_ + " ")
    name = re.sub(r"[-_]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def normalize_for_matching(name: str) -> str:
    """Aggressive normalization that strips ALL non-alphanumeric characters.

    This ensures that entity IDs (``homair``), display names (``HOMA-IR``),
    and snake_case IDs (``homa_ir``) all collapse to the same key
    (``homair``).  Use this when comparing entity names across boundaries
    where one side uses IDs and the other uses labels/display names.

    Examples::

        normalize_for_matching("HOMA-IR")          -> "homair"
        normalize_for_matching("homair")            -> "homair"
        normalize_for_matching("homa_ir")           -> "homair"
        normalize_for_matching("IL-6")              -> "il6"
        normalize_for_matching("il_6")              -> "il6"
        normalize_for_matching("TNF-α")             -> "tnfalpha"
        normalize_for_matching("insulin resistance") -> "insulinresistance"
        normalize_for_matching("insulin_resistance") -> "insulinresistance"
    """
    name = name.lower().strip()
    for greek, ascii_ in _GREEK_TO_ASCII.items():
        name = name.replace(greek, ascii_)
    return re.sub(r"[^a-z0-9]", "", name)


def fuzzy_entity_match(
    a: str,
    b: str,
    threshold: float = 0.85,
) -> Tuple[bool, float]:
    """
    Return (is_match, similarity) for two entity names using
    normalized SequenceMatcher ratio.
    """
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True, 1.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return ratio >= threshold, ratio


def find_entity_matches(
    source_entities: Dict[str, Entity],
    target_entities: Dict[str, Entity],
    threshold: float = 0.80,
) -> Dict[str, str]:
    """
    Build a mapping {source_id -> target_id} for entities that likely
    refer to the same real-world concept.

    Checks:
    1. Exact id match
    2. Label overlap
    3. Fuzzy name similarity (SequenceMatcher)
    """
    mapping: Dict[str, str] = {}
    target_name_index: Dict[str, str] = {}

    # Build reverse-index: normalized name/label -> target_id
    for tid, tentity in target_entities.items():
        target_name_index[_normalize(tid)] = tid
        for label in tentity.labels:
            target_name_index[_normalize(label)] = tid

    for sid, sentity in source_entities.items():
        # 1. Exact id match
        if sid in target_entities:
            mapping[sid] = sid
            continue

        # 2. Label / name index lookup
        norm_sid = _normalize(sid)
        if norm_sid in target_name_index:
            mapping[sid] = target_name_index[norm_sid]
            continue

        label_matched = False
        for label in sentity.labels:
            nlabel = _normalize(label)
            if nlabel in target_name_index:
                mapping[sid] = target_name_index[nlabel]
                label_matched = True
                break
        if label_matched:
            continue

        # 3. Fuzzy matching against all target names
        best_score, best_tid = 0.0, None
        for tname, tid in target_name_index.items():
            is_match, score = fuzzy_entity_match(norm_sid, tname, threshold)
            if is_match and score > best_score:
                best_score, best_tid = score, tid
        if best_tid:
            mapping[sid] = best_tid

    return mapping


# ---------------------------------------------------------------------------
# KG Diff
# ---------------------------------------------------------------------------

@dataclass
class KGDiff:
    """Represents the difference between two knowledge graphs."""

    new_entities: List[Entity] = field(default_factory=list)
    updated_entities: List[Tuple[Entity, Entity]] = field(default_factory=list)  # (old, new)
    new_triples: List[Triple] = field(default_factory=list)
    conflicting_triples: List[Tuple[Triple, Triple]] = field(default_factory=list)  # (existing, candidate)
    entity_mapping: Dict[str, str] = field(default_factory=dict)  # new_id -> existing_id

    @property
    def has_changes(self) -> bool:
        return bool(
            self.new_entities
            or self.updated_entities
            or self.new_triples
            or self.conflicting_triples
        )

    def summary(self) -> str:
        lines = ["--- KG Diff ---"]
        lines.append(f"  New entities:        {len(self.new_entities)}")
        lines.append(f"  Updated entities:    {len(self.updated_entities)}")
        lines.append(f"  New triples:         {len(self.new_triples)}")
        lines.append(f"  Conflicting triples: {len(self.conflicting_triples)}")
        lines.append(f"  Entity mappings:     {len(self.entity_mapping)}")
        return "\n".join(lines)


def compute_diff(
    base_kg: KnowledgeGraph,
    delta_kg: KnowledgeGraph,
    match_threshold: float = 0.80,
) -> KGDiff:
    """
    Compute the diff between a base KG and a delta (incoming) KG.

    Returns a KGDiff describing what would change if delta were merged.
    """
    diff = KGDiff()

    # Entity resolution
    diff.entity_mapping = find_entity_matches(
        delta_kg.entities, base_kg.entities, match_threshold
    )

    for eid, entity in delta_kg.entities.items():
        if eid in diff.entity_mapping:
            existing = base_kg.entities[diff.entity_mapping[eid]]
            # Check if there is anything new to merge
            new_labels = [l for l in entity.labels if l not in existing.labels]
            new_meta = {
                k: v
                for k, v in entity.metadata.items()
                if k not in existing.metadata
            }
            if new_labels or new_meta or (entity.type and not existing.type):
                diff.updated_entities.append((existing, entity))
        else:
            diff.new_entities.append(entity)

    # Triple resolution (after remapping entity ids)
    existing_triple_keys: Set[Tuple[str, str, str]] = set()
    existing_triple_map: Dict[Tuple[str, str], List[Triple]] = {}
    for t in base_kg.triples:
        existing_triple_keys.add((t.subject, t.relation, t.object))
        key = (t.subject, t.relation)
        existing_triple_map.setdefault(key, []).append(t)

    for t in delta_kg.triples:
        # Remap entity ids
        subj = diff.entity_mapping.get(t.subject, t.subject)
        obj = diff.entity_mapping.get(t.object, t.object)
        rel = t.relation

        if (subj, rel, obj) in existing_triple_keys:
            continue  # Already exists

        # Check for conflicts (same subject+relation, different object)
        key = (subj, rel)
        if key in existing_triple_map:
            for existing_t in existing_triple_map[key]:
                if existing_t.object != obj:
                    remapped = Triple(
                        subject=subj,
                        relation=rel,
                        object=obj,
                        confidence=t.confidence,
                        source=t.source,
                        metadata=t.metadata,
                    )
                    diff.conflicting_triples.append((existing_t, remapped))
        else:
            diff.new_triples.append(
                Triple(
                    subject=subj,
                    relation=rel,
                    object=obj,
                    confidence=t.confidence,
                    source=t.source,
                    metadata=t.metadata,
                )
            )

    return diff


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_kg(
    base_kg: KnowledgeGraph,
    diff: KGDiff,
    conflict_strategy: str = "keep_higher_confidence",
) -> Dict[str, Any]:
    """
    Apply a KGDiff to a base KG in-place.

    Args:
        base_kg: The target knowledge graph (mutated in place).
        diff: The diff to apply.
        conflict_strategy: One of "keep_higher_confidence", "keep_existing", "keep_new", "keep_both".

    Returns:
        Statistics dict about the merge operation.
    """
    stats = {
        "entities_added": 0,
        "entities_updated": 0,
        "triples_added": 0,
        "conflicts_resolved": 0,
    }

    # Add new entities
    for entity in diff.new_entities:
        base_kg.add_entity(
            entity_id=entity.id,
            labels=entity.labels,
            entity_type=entity.type,
            metadata=entity.metadata,
        )
        stats["entities_added"] += 1

    # Update existing entities
    for existing, incoming in diff.updated_entities:
        e = base_kg.entities[existing.id]
        for label in incoming.labels:
            if label not in e.labels:
                e.labels.append(label)
        if incoming.type and not e.type:
            e.type = incoming.type
        e.metadata.update(incoming.metadata)
        stats["entities_updated"] += 1

    # Add new triples
    for triple in diff.new_triples:
        base_kg.add_triple(
            subject=triple.subject,
            relation=triple.relation,
            obj=triple.object,
            confidence=triple.confidence,
            source=triple.source,
            metadata=triple.metadata,
        )
        stats["triples_added"] += 1

    # Resolve conflicts
    for existing_t, candidate_t in diff.conflicting_triples:
        if conflict_strategy == "keep_existing":
            pass
        elif conflict_strategy == "keep_new":
            # Remove existing, add new
            base_kg.triples = [
                t for t in base_kg.triples if t != existing_t
            ]
            base_kg._triple_set.discard(existing_t)
            base_kg.add_triple(
                subject=candidate_t.subject,
                relation=candidate_t.relation,
                obj=candidate_t.object,
                confidence=candidate_t.confidence,
                source=candidate_t.source,
                metadata=candidate_t.metadata,
            )
        elif conflict_strategy == "keep_higher_confidence":
            ec = existing_t.confidence or 0.0
            cc = candidate_t.confidence or 0.0
            if cc > ec:
                base_kg.triples = [
                    t for t in base_kg.triples if t != existing_t
                ]
                base_kg._triple_set.discard(existing_t)
                base_kg.add_triple(
                    subject=candidate_t.subject,
                    relation=candidate_t.relation,
                    obj=candidate_t.object,
                    confidence=candidate_t.confidence,
                    source=candidate_t.source,
                    metadata=candidate_t.metadata,
                )
        elif conflict_strategy == "keep_both":
            base_kg.add_triple(
                subject=candidate_t.subject,
                relation=candidate_t.relation,
                obj=candidate_t.object,
                confidence=candidate_t.confidence,
                source=candidate_t.source,
                metadata=candidate_t.metadata,
            )
        stats["conflicts_resolved"] += 1

    return stats


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_kg(kg: KnowledgeGraph, path: str) -> None:
    """Save a KG to a JSON file."""
    data = kg.to_dict()
    data["stats"] = kg.get_stats()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_kg(path: str) -> KnowledgeGraph:
    """Load a KG from a previously-exported JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kg = KnowledgeGraph()

    entities_list = data.get("entities", [])
    if isinstance(data, dict) and "knowledge_graph" in data:
        entities_list = data["knowledge_graph"].get("entities", [])
        triples_list = data["knowledge_graph"].get("triples", [])
    else:
        triples_list = data.get("triples", [])

    for e in entities_list:
        kg.add_entity(
            entity_id=e["id"],
            labels=e.get("labels", []),
            entity_type=e.get("type"),
            metadata=e.get("metadata", {}),
        )

    for t in triples_list:
        subj = t.get("subject", "")
        rel = t.get("relation", "")
        obj = t.get("object", "")
        if subj and rel and obj:
            kg.add_triple(
                subject=subj,
                relation=rel,
                obj=obj,
                confidence=t.get("confidence"),
                source=t.get("source"),
                metadata=t.get("metadata", {}),
            )

    return kg


def save_governed_kg(gkg: GovernedKnowledgeGraph, path: str) -> None:
    """Save a governed knowledge graph to JSON."""
    data = gkg.to_dict()
    data["stats"] = gkg.get_stats()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def load_governed_kg(path: str) -> GovernedKnowledgeGraph:
    """Load a governed KG, auto-wrapping legacy plain KG exports."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return GovernedKnowledgeGraph.from_dict(data)
