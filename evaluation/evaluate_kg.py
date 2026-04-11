#!/usr/bin/env python3
"""
Evaluate a Knowledge Graph extraction pipeline against gold-standard data.

Supports two input modes:

1. **Per-document evaluation** (SciERC style):
   --gold file contains a JSON array of gold documents, each with
   "doc_key", "entities", and "triples".
   --predicted is a directory of per-document kg_export JSON files,
   named <doc_key>.json.

2. **Single-file evaluation**:
   --gold is a single gold JSON (same format as above, one doc).
   --predicted is a single kg_export.json from the pipeline.

Metrics computed:
  - Entity extraction: Precision / Recall / F1  (strict and partial match)
  - Relation extraction: Triple-level P / R / F1  (strict and fuzzy)
  - Entity type accuracy (of correctly matched entities)
  - Hallucination rate (extracted items with no gold match)

Usage:
    python evaluation/evaluate_kg.py \\
        --gold evaluation/datasets/scierc/test.json \\
        --predicted kg_export.json \\
        [--fuzzy-threshold 0.8] [--format scierc|export]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# String-similarity helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation edges."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?()[]{}\"'")
    return text


def _fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """Return True if normalised texts are similar above *threshold*."""
    na, nb = _normalise_text(a), _normalise_text(b)
    if na == nb:
        return True
    ratio = SequenceMatcher(None, na, nb).ratio()
    return ratio >= threshold


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    ta = set(_normalise_text(a).split())
    tb = set(_normalise_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _entity_surface_text(entity: Dict[str, Any]) -> str:
    """Return the best human-readable surface form for an entity."""
    text = entity.get("text")
    if text:
        return text
    labels = entity.get("labels", [])
    if labels:
        return labels[0]
    return entity.get("id", "")


def _triple_surface_variants(triple: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return plausible surface-form variants for a triple's subject/object.

    Governed KG exports often use canonical IDs in `subject`/`object` while
    preserving the original mention text in metadata. The evaluator should
    score those exports against gold text on the same footing as older
    per-document exports that stored surface strings directly.
    """
    metadata = triple.get("metadata", {}) or {}
    subjects: List[str] = []
    objects: List[str] = []

    for value in [
        triple.get("subject", ""),
        metadata.get("original_subject", ""),
        metadata.get("subject_text", ""),
    ]:
        if value and value not in subjects:
            subjects.append(value)

    for value in [
        triple.get("object", ""),
        metadata.get("original_object", ""),
        metadata.get("object_text", ""),
    ]:
        if value and value not in objects:
            objects.append(value)

    if not subjects:
        subjects.append("")
    if not objects:
        objects.append("")

    return [(subj, obj) for subj in subjects for obj in objects]


# ---------------------------------------------------------------------------
# SciERC type/relation mapping for dynamic→fixed schema evaluation
# ---------------------------------------------------------------------------

# Maps our dynamic entity types to SciERC's 5 canonical types
ENTITY_TYPE_MAP: Dict[str, str] = {
    # → Task
    "TASK": "Task", "ANALYSIS_STAGE": "Task", "NLP_TASK": "Task",
    "RESEARCH_TASK": "Task", "CLASSIFICATION_TASK": "Task",
    "RECOGNITION_TASK": "Task", "PROCESSING_TASK": "Task",
    "EVALUATION_TASK": "Task", "LEARNING_TASK": "Task",
    # → Method
    "METHOD": "Method", "ALGORITHM": "Method", "MODEL": "Method",
    "TECHNIQUE": "Method", "APPROACH": "Method", "FRAMEWORK": "Method",
    "SYSTEM": "Method", "ARCHITECTURE": "Method", "CLASSIFIER": "Method",
    "NEURAL_NETWORK": "Method", "TOOL": "Method", "PIPELINE": "Method",
    "COMPUTATIONAL_METHOD": "Method", "OPTIMIZATION_METHOD": "Method",
    # → Metric
    "METRIC": "Metric", "EVALUATION_METRIC": "Metric",
    "PERFORMANCE_METRIC": "Metric", "MEASUREMENT": "Metric",
    "SCORE": "Metric", "STATISTICAL_MEASURE": "Metric",
    # → Material
    "MATERIAL": "Material", "DATASET": "Material", "CORPUS": "Material",
    "DATA": "Material", "RESOURCE": "Material", "LANGUAGE": "Material",
    "TEXT_CORPUS": "Material", "BENCHMARK": "Material",
    "TRAINING_DATA": "Material", "INPUT_DATA": "Material",
    # → OtherScientificTerm
    "OTHERSCIENTIFICTERM": "OtherScientificTerm",
    "CONCEPT": "OtherScientificTerm", "DOMAIN_CONCEPT": "OtherScientificTerm",
    "SCIENTIFIC_TERM": "OtherScientificTerm",
    "MATHEMATICAL_CONCEPT": "OtherScientificTerm",
    "LINGUISTIC_CONCEPT": "OtherScientificTerm",
    "SYNTACTIC_CONCEPT": "OtherScientificTerm",
    "FEATURE": "OtherScientificTerm", "REPRESENTATION": "OtherScientificTerm",
    "PARAMETER": "OtherScientificTerm", "CONSTRAINT": "OtherScientificTerm",
    "GEOMETRIC_CONSTRAINT": "OtherScientificTerm",
    "STRUCTURE": "OtherScientificTerm", "PROPERTY": "OtherScientificTerm",
    "DEPENDENCY_TREE": "OtherScientificTerm",
    "STATISTICAL_CONCEPT": "OtherScientificTerm",
    "NAMED_ENTITY_ITEM": "OtherScientificTerm",
}

# Maps our dynamic relation types to SciERC's 7 canonical relation types
RELATION_TYPE_MAP: Dict[str, str] = {
    # → Used-for
    "USED_FOR": "Used-for", "USES": "Used-for", "UTILIZES": "Used-for",
    "APPLIED_TO": "Used-for", "APPLIES_TO": "Used-for",
    "UTILIZES_RESOURCE": "Used-for", "USES_AS_CONSTRAINT": "Used-for",
    "PROCESSES_TEXT": "Used-for", "PERFORMS_FUNCTION": "Used-for",
    "VALIDATED_ON_DATA": "Used-for", "STUDIED_IN_FRAMEWORK": "Used-for",
    "BASED_ON": "Used-for",
    # → Feature-of
    "FEATURE_OF": "Feature-of", "HAS_FEATURE": "Feature-of",
    "PROPERTY_OF": "Feature-of", "ATTRIBUTE_OF": "Feature-of",
    "CHARACTERIZES": "Feature-of",
    # → Part-of
    "PART_OF": "Part-of", "IS_PART_OF": "Part-of",
    "COMPONENT_OF": "Part-of", "CONTAINS": "Part-of",
    "HAS_COMPONENT": "Part-of", "SUBSET_OF": "Part-of",
    "INCORPORATES_CONSTRAINT": "Part-of",
    # → Compare
    "COMPARE": "Compare", "COMPARED_TO": "Compare",
    "OUTPERFORMS": "Compare", "REPLACES_RESOURCE": "Compare",
    "ADDRESSES_LIMITATION": "Compare",
    # → Hyponym-of
    "HYPONYM_OF": "Hyponym-of", "IS_A": "Hyponym-of",
    "TYPE_OF": "Hyponym-of", "SUBTYPE_OF": "Hyponym-of",
    "INSTANCE_OF": "Hyponym-of",
    # → Conjunction
    "CONJUNCTION": "Conjunction", "AND": "Conjunction",
    "COMBINED_WITH": "Conjunction", "RELATES_WORDS": "Conjunction",
    # → Evaluate-for
    "EVALUATE_FOR": "Evaluate-for", "EVALUATES": "Evaluate-for",
    "MAXIMIZES_MEASUREMENT": "Evaluate-for",
    "MINIMIZES_FUNCTION": "Evaluate-for",
    # Catch-all for verb-like relations → Used-for (most common SciERC type)
    "ENABLES_RESULT": "Used-for", "IDENTIFIES_ITEM": "Used-for",
    "ADDS_INFORMATION": "Part-of", "CREATES_STRUCTURE": "Used-for",
    "ENCODES_PREFERENCE": "Feature-of", "ENCODES_KNOWLEDGE": "Feature-of",
    "ENFORCES_CONSISTENCY": "Feature-of", "DEFORMS_MESH": "Used-for",
    "MODELS_CONCEPT": "Used-for", "SAMPLES_ENTITIES": "Used-for",
}


def _map_entity_type(etype: str) -> str:
    """Map a dynamic entity type to a SciERC canonical type."""
    upper = etype.upper().replace(" ", "_").replace("-", "_")
    if upper in ENTITY_TYPE_MAP:
        return ENTITY_TYPE_MAP[upper]
    # Heuristic: if it contains keywords
    low = etype.lower()
    if any(w in low for w in ("task", "recognition", "classification", "detection")):
        return "Task"
    if any(w in low for w in ("method", "algorithm", "model", "network", "system")):
        return "Method"
    if any(w in low for w in ("metric", "score", "accuracy", "precision", "f1")):
        return "Metric"
    if any(w in low for w in ("data", "corpus", "dataset", "language", "text")):
        return "Material"
    return "OtherScientificTerm"


def _map_relation_type(rtype: str) -> str:
    """Map a dynamic relation type to a SciERC canonical relation type."""
    upper = rtype.upper().replace(" ", "_").replace("-", "_")
    if upper in RELATION_TYPE_MAP:
        return RELATION_TYPE_MAP[upper]
    # Check with hyphens removed
    no_hyphen = rtype.upper().replace("-", "_")
    if no_hyphen in RELATION_TYPE_MAP:
        return RELATION_TYPE_MAP[no_hyphen]
    # Default: Used-for is the most common
    return "Used-for"


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------

class PRF:
    """Precision / Recall / F1 accumulator."""

    def __init__(self) -> None:
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __repr__(self) -> str:
        return (
            f"P={self.precision():.3f}  R={self.recall():.3f}  "
            f"F1={self.f1():.3f}  (TP={self.tp} FP={self.fp} FN={self.fn})"
        )


# ---------------------------------------------------------------------------
# Gold / predicted loaders
# ---------------------------------------------------------------------------

def load_gold_scierc(path: str) -> List[Dict[str, Any]]:
    """
    Load gold-standard data.

    Supports:
      - SciERC JSON-lines (one JSON object per line with sentences/ner/relations)
      - Pre-converted JSON array (output of scierc_adapter --output-gold)
    """
    # Try JSON array first
    with open(path, "r", encoding="utf-8") as fh:
        first_char = fh.read(1)

    if first_char == "[":
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Otherwise treat as SciERC JSON-lines -> convert on the fly
    # Import the adapter
    adapter_dir = os.path.join(os.path.dirname(__file__), "adapters")
    sys.path.insert(0, adapter_dir)
    from scierc_adapter import SciERCAdapter
    adapter = SciERCAdapter(path)
    return adapter.get_all_gold()


def load_predicted_single(path: str) -> Dict[str, Any]:
    """
    Load a single pipeline kg_export.json.

    Expected format:
        {"knowledge_graph": {"entities": [...], "triples": [...]}}
    or just:
        {"entities": [...], "triples": [...]}
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "knowledge_graph" in data:
        return data["knowledge_graph"]
    return data


def load_predicted_dir(dirpath: str) -> Dict[str, Dict[str, Any]]:
    """Load per-document predicted results from a directory of JSON files.
    Returns {doc_key: {"entities": [...], "triples": [...]}}.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for fname in os.listdir(dirpath):
        if not fname.endswith(".json"):
            continue
        doc_key = fname.replace(".json", "")
        results[doc_key] = load_predicted_single(os.path.join(dirpath, fname))
    return results


# ---------------------------------------------------------------------------
# Entity evaluation
# ---------------------------------------------------------------------------

def evaluate_entities_strict(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    map_types: bool = False,
) -> Tuple[PRF, Dict[str, PRF], float, float]:
    """
    Strict entity matching: exact normalised text must match.

    Args:
        map_types: If True, map predicted entity types to SciERC canonical
                   types before comparing.

    Returns:
        overall_prf: aggregate PRF
        per_type_prf: dict mapping entity type -> PRF
        type_accuracy: fraction of matched entities with correct type
        hallucination_rate: fraction of predicted entities with no gold match
    """
    prf = PRF()
    per_type: Dict[str, PRF] = defaultdict(PRF)

    # Build gold set  (text -> type)
    gold_set: Dict[str, str] = {}
    for ent in gold_entities:
        gold_set[_normalise_text(ent["text"])] = ent["type"]

    # Build predicted set — include all label variants for matching
    pred_set: Dict[str, str] = {}
    for ent in pred_entities:
        text = _entity_surface_text(ent)
        etype = ent.get("type", "Unknown")
        if map_types:
            etype = _map_entity_type(etype)
        pred_set[_normalise_text(text)] = etype
        # Also add all label forms as matchable entries
        for label in ent.get("labels", []):
            norm_label = _normalise_text(label)
            if norm_label and norm_label not in pred_set:
                pred_set[norm_label] = etype

    matched_correct_type = 0
    matched_total = 0

    gold_keys = set(gold_set.keys())
    pred_keys = set(pred_set.keys())

    matched = gold_keys & pred_keys
    prf.tp = len(matched)
    prf.fp = len(pred_keys - gold_keys)
    prf.fn = len(gold_keys - pred_keys)

    # Per-type and type accuracy
    for key in matched:
        gtype = gold_set[key]
        ptype = pred_set[key]
        per_type[gtype].tp += 1
        matched_total += 1
        if _normalise_text(gtype) == _normalise_text(ptype):
            matched_correct_type += 1

    for key in gold_keys - pred_keys:
        gtype = gold_set[key]
        per_type[gtype].fn += 1

    for key in pred_keys - gold_keys:
        ptype = pred_set[key]
        per_type[ptype].fp += 1

    type_acc = matched_correct_type / matched_total if matched_total else 0.0
    halluc = prf.fp / len(pred_keys) if pred_keys else 0.0

    return prf, dict(per_type), type_acc, halluc


def evaluate_entities_partial(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> Tuple[PRF, float]:
    """
    Partial entity matching using token overlap (Jaccard >= threshold).

    Returns:
        prf: aggregate PRF
        hallucination_rate: fraction of predicted with no partial match
    """
    prf = PRF()

    gold_texts = [_normalise_text(e["text"]) for e in gold_entities]
    pred_texts = [
        _normalise_text(_entity_surface_text(e))
        for e in pred_entities
    ]

    gold_matched: Set[int] = set()
    pred_matched: Set[int] = set()

    for pi, pt in enumerate(pred_texts):
        best_overlap = 0.0
        best_gi = -1
        for gi, gt in enumerate(gold_texts):
            if gi in gold_matched:
                continue
            ov = _token_overlap(pt, gt)
            if ov > best_overlap:
                best_overlap = ov
                best_gi = gi
        if best_overlap >= overlap_threshold and best_gi >= 0:
            prf.tp += 1
            gold_matched.add(best_gi)
            pred_matched.add(pi)

    prf.fp = len(pred_texts) - len(pred_matched)
    prf.fn = len(gold_texts) - len(gold_matched)

    halluc = prf.fp / len(pred_texts) if pred_texts else 0.0
    return prf, halluc


# ---------------------------------------------------------------------------
# Relation / triple evaluation
# ---------------------------------------------------------------------------

def _triple_key_strict(t: Dict[str, Any]) -> Tuple[str, str, str]:
    subj = _normalise_text(t.get("subject", ""))
    rel = _normalise_text(t.get("relation", ""))
    obj = _normalise_text(t.get("object", ""))
    return (subj, rel, obj)


def evaluate_triples_strict(
    gold_triples: List[Dict[str, Any]],
    pred_triples: List[Dict[str, Any]],
) -> Tuple[PRF, Dict[str, PRF], float]:
    """
    Strict triple matching: normalised (subject, relation, object) must match exactly.

    Returns:
        overall_prf, per_relation_prf, hallucination_rate
    """
    prf = PRF()
    per_rel: Dict[str, PRF] = defaultdict(PRF)
    gold_matched: Set[int] = set()
    pred_matched: Set[int] = set()

    for pi, pt in enumerate(pred_triples):
        p_rel = _normalise_text(pt.get("relation", ""))
        matched = False
        for gi, gt in enumerate(gold_triples):
            if gi in gold_matched:
                continue
            g_rel = _normalise_text(gt.get("relation", ""))
            if p_rel != g_rel:
                continue
            g_key = _triple_key_strict(gt)
            for subj, obj in _triple_surface_variants(pt):
                p_key = (
                    _normalise_text(subj),
                    p_rel,
                    _normalise_text(obj),
                )
                if p_key == g_key:
                    prf.tp += 1
                    gold_matched.add(gi)
                    pred_matched.add(pi)
                    per_rel[g_rel].tp += 1
                    matched = True
                    break
            if matched:
                break

    prf.fp = len(pred_triples) - len(pred_matched)
    prf.fn = len(gold_triples) - len(gold_matched)

    for gi, gt in enumerate(gold_triples):
        if gi not in gold_matched:
            per_rel[_normalise_text(gt.get("relation", ""))].fn += 1
    for pi, pt in enumerate(pred_triples):
        if pi not in pred_matched:
            per_rel[_normalise_text(pt.get("relation", ""))].fp += 1

    halluc = prf.fp / len(pred_triples) if pred_triples else 0.0
    return prf, dict(per_rel), halluc


def evaluate_triples_fuzzy(
    gold_triples: List[Dict[str, Any]],
    pred_triples: List[Dict[str, Any]],
    threshold: float = 0.8,
) -> Tuple[PRF, float]:
    """
    Fuzzy triple matching: subject and object are fuzzy-matched, relation
    must match exactly (normalised).

    Returns:
        prf, hallucination_rate
    """
    prf = PRF()
    gold_matched: Set[int] = set()
    pred_matched: Set[int] = set()

    for pi, pt in enumerate(pred_triples):
        p_rel = _normalise_text(pt.get("relation", ""))
        for gi, gt in enumerate(gold_triples):
            if gi in gold_matched:
                continue
            g_rel = _normalise_text(gt.get("relation", ""))
            if p_rel != g_rel:
                continue
            for p_subj, p_obj in _triple_surface_variants(pt):
                if _fuzzy_match(p_subj, gt["subject"], threshold) and \
                   _fuzzy_match(p_obj, gt["object"], threshold):
                    prf.tp += 1
                    gold_matched.add(gi)
                    pred_matched.add(pi)
                    break
            if pi in pred_matched:
                break

    prf.fp = len(pred_triples) - len(pred_matched)
    prf.fn = len(gold_triples) - len(gold_matched)

    halluc = prf.fp / len(pred_triples) if pred_triples else 0.0
    return prf, halluc


def evaluate_triples_mapped(
    gold_triples: List[Dict[str, Any]],
    pred_triples: List[Dict[str, Any]],
    threshold: float = 0.8,
) -> Tuple[PRF, float]:
    """
    Mapped triple matching: map predicted relation types to SciERC canonical
    types, then fuzzy-match subject/object.

    Returns:
        prf, hallucination_rate
    """
    prf = PRF()
    gold_matched: Set[int] = set()
    pred_matched: Set[int] = set()

    for pi, pt in enumerate(pred_triples):
        p_rel_raw = pt.get("relation", "")
        p_rel = _normalise_text(_map_relation_type(p_rel_raw))
        for gi, gt in enumerate(gold_triples):
            if gi in gold_matched:
                continue
            g_rel = _normalise_text(gt.get("relation", ""))
            if p_rel != g_rel:
                continue
            for p_subj, p_obj in _triple_surface_variants(pt):
                if _fuzzy_match(p_subj, gt["subject"], threshold) and \
                   _fuzzy_match(p_obj, gt["object"], threshold):
                    prf.tp += 1
                    gold_matched.add(gi)
                    pred_matched.add(pi)
                    break
            if pi in pred_matched:
                break

    prf.fp = len(pred_triples) - len(pred_matched)
    prf.fn = len(gold_triples) - len(gold_matched)

    halluc = prf.fp / len(pred_triples) if pred_triples else 0.0
    return prf, halluc


# ---------------------------------------------------------------------------
# Aggregate evaluation over multiple documents
# ---------------------------------------------------------------------------

def evaluate_corpus(
    gold_docs: List[Dict[str, Any]],
    pred_docs: Dict[str, Dict[str, Any]],
    fuzzy_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Evaluate all predicted documents against gold standard.

    Args:
        gold_docs: list of gold docs (from adapter or pre-converted JSON).
        pred_docs: mapping doc_key -> {"entities": [...], "triples": [...]}.
        fuzzy_threshold: similarity threshold for fuzzy matching.

    Returns:
        Dictionary of all computed metrics.
    """
    # Accumulators
    ent_strict = PRF()
    ent_strict_mapped = PRF()
    ent_partial = PRF()
    tri_strict = PRF()
    tri_fuzzy = PRF()
    tri_mapped = PRF()
    per_ent_type: Dict[str, PRF] = defaultdict(PRF)
    per_rel_type: Dict[str, PRF] = defaultdict(PRF)
    type_correct = 0
    type_total = 0
    type_correct_mapped = 0
    type_total_mapped = 0
    total_pred_ents = 0
    total_pred_trips = 0
    ent_halluc_count = 0
    tri_halluc_count = 0
    docs_evaluated = 0
    docs_missing = 0

    for gdoc in gold_docs:
        doc_key = gdoc["doc_key"]
        if doc_key not in pred_docs:
            docs_missing += 1
            # All gold items are false negatives
            g_ents = gdoc["entities"]
            g_trips = gdoc["triples"]
            ent_strict.fn += len(g_ents)
            ent_partial.fn += len(g_ents)
            tri_strict.fn += len(g_trips)
            tri_fuzzy.fn += len(g_trips)
            for e in g_ents:
                per_ent_type[e["type"]].fn += 1
            for t in g_trips:
                per_rel_type[_normalise_text(t["relation"])].fn += 1
            continue

        docs_evaluated += 1
        pred = pred_docs[doc_key]
        g_ents = gdoc["entities"]
        g_trips = gdoc["triples"]
        p_ents = pred.get("entities", [])
        p_trips = pred.get("triples", [])
        total_pred_ents += len(p_ents)
        total_pred_trips += len(p_trips)

        # Entity strict (raw types)
        es, es_pt, ta, eh = evaluate_entities_strict(g_ents, p_ents)
        ent_strict.tp += es.tp
        ent_strict.fp += es.fp
        ent_strict.fn += es.fn
        type_correct += int(ta * (es.tp if es.tp else 0))
        type_total += es.tp
        ent_halluc_count += es.fp
        for t, prf in es_pt.items():
            per_ent_type[t].tp += prf.tp
            per_ent_type[t].fp += prf.fp
            per_ent_type[t].fn += prf.fn

        # Entity strict (mapped types)
        es_m, es_m_pt, ta_m, _ = evaluate_entities_strict(
            g_ents, p_ents, map_types=True
        )
        ent_strict_mapped.tp += es_m.tp
        ent_strict_mapped.fp += es_m.fp
        ent_strict_mapped.fn += es_m.fn
        type_correct_mapped += int(ta_m * (es_m.tp if es_m.tp else 0))
        type_total_mapped += es_m.tp

        # Entity partial
        ep, _ = evaluate_entities_partial(g_ents, p_ents)
        ent_partial.tp += ep.tp
        ent_partial.fp += ep.fp
        ent_partial.fn += ep.fn

        # Triple strict
        ts, ts_pr, _ = evaluate_triples_strict(g_trips, p_trips)
        tri_strict.tp += ts.tp
        tri_strict.fp += ts.fp
        tri_strict.fn += ts.fn
        tri_halluc_count += ts.fp
        for r, prf in ts_pr.items():
            per_rel_type[r].tp += prf.tp
            per_rel_type[r].fp += prf.fp
            per_rel_type[r].fn += prf.fn

        # Triple fuzzy
        tf, _ = evaluate_triples_fuzzy(g_trips, p_trips, fuzzy_threshold)
        tri_fuzzy.tp += tf.tp
        tri_fuzzy.fp += tf.fp
        tri_fuzzy.fn += tf.fn

        # Triple mapped (relation types mapped to SciERC canonical)
        tm, _ = evaluate_triples_mapped(g_trips, p_trips, fuzzy_threshold)
        tri_mapped.tp += tm.tp
        tri_mapped.fp += tm.fp
        tri_mapped.fn += tm.fn

    return {
        "documents_total": len(gold_docs),
        "documents_evaluated": docs_evaluated,
        "documents_missing_predictions": docs_missing,
        "entity_strict": ent_strict,
        "entity_strict_mapped": ent_strict_mapped,
        "entity_partial": ent_partial,
        "entity_per_type": dict(per_ent_type),
        "entity_type_accuracy": type_correct / type_total if type_total else 0.0,
        "entity_type_accuracy_mapped": (
            type_correct_mapped / type_total_mapped if type_total_mapped else 0.0
        ),
        "entity_hallucination_rate": (
            ent_halluc_count / total_pred_ents if total_pred_ents else 0.0
        ),
        "triple_strict": tri_strict,
        "triple_fuzzy": tri_fuzzy,
        "triple_mapped": tri_mapped,
        "triple_per_relation": dict(per_rel_type),
        "triple_hallucination_rate": (
            tri_halluc_count / total_pred_trips if total_pred_trips else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(metrics: Dict[str, Any]) -> None:
    """Print a clean evaluation report."""
    sep = "=" * 72
    thin = "-" * 72

    print(sep)
    print("  KNOWLEDGE GRAPH EVALUATION REPORT")
    print(sep)
    print()
    print(f"  Documents total:              {metrics['documents_total']}")
    print(f"  Documents evaluated:          {metrics['documents_evaluated']}")
    print(f"  Documents missing predictions:{metrics['documents_missing_predictions']}")
    print()

    # -- Entity metrics --
    print(thin)
    print("  ENTITY EXTRACTION")
    print(thin)
    es: PRF = metrics["entity_strict"]
    ep: PRF = metrics["entity_partial"]
    print(f"  Strict match:   {es}")
    print(f"  Partial match:  {ep}")
    print(f"  Type accuracy (raw):    {metrics['entity_type_accuracy']:.3f}")
    if "entity_strict_mapped" in metrics:
        esm: PRF = metrics["entity_strict_mapped"]
        print(f"  Strict (mapped types):  {esm}")
        print(f"  Type accuracy (mapped): {metrics.get('entity_type_accuracy_mapped', 0):.3f}")
    print(f"  Halluc. rate:   {metrics['entity_hallucination_rate']:.3f}")
    print()

    per_type = metrics["entity_per_type"]
    if per_type:
        print("  Per-type breakdown (strict):")
        for etype in sorted(per_type.keys()):
            p = per_type[etype]
            print(f"    {etype:30s}  {p}")
    print()

    # -- Triple metrics --
    print(thin)
    print("  RELATION EXTRACTION (TRIPLES)")
    print(thin)
    ts: PRF = metrics["triple_strict"]
    tf: PRF = metrics["triple_fuzzy"]
    print(f"  Strict match:   {ts}")
    print(f"  Fuzzy match:    {tf}")
    if "triple_mapped" in metrics:
        tm: PRF = metrics["triple_mapped"]
        print(f"  Mapped match:   {tm}")
    print(f"  Halluc. rate:   {metrics['triple_hallucination_rate']:.3f}")
    print()

    per_rel = metrics["triple_per_relation"]
    if per_rel:
        print("  Per-relation breakdown (strict):")
        for rtype in sorted(per_rel.keys()):
            p = per_rel[rtype]
            print(f"    {rtype:30s}  {p}")
    print()
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KG extraction against gold standard."
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Path to gold-standard file (SciERC JSON-lines or converted JSON array)",
    )
    parser.add_argument(
        "--predicted",
        required=True,
        help="Path to predicted kg_export.json file or directory of per-doc JSONs",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy matching (default: 0.8)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit evaluation to first N gold documents",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write metrics to JSON file",
    )
    args = parser.parse_args()

    # --- Load gold ---
    print(f"Loading gold standard from: {args.gold}")
    gold_docs = load_gold_scierc(args.gold)
    if args.max_docs:
        gold_docs = gold_docs[: args.max_docs]
    print(f"  {len(gold_docs)} documents loaded")

    # --- Load predictions ---
    print(f"Loading predictions from: {args.predicted}")
    if os.path.isdir(args.predicted):
        pred_docs = load_predicted_dir(args.predicted)
    else:
        # Single file -- try to map to gold doc keys
        pred_kg = load_predicted_single(args.predicted)
        # If there's only one gold doc, map it
        if len(gold_docs) == 1:
            pred_docs = {gold_docs[0]["doc_key"]: pred_kg}
        else:
            # Assume all predictions belong to a single "corpus" document
            # or try to use doc_key field if present
            pred_docs = {}
            # Check if the file contains per-doc results
            with open(args.predicted, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                # Array of per-doc results
                for item in raw:
                    dk = item.get("doc_key", "")
                    if dk:
                        kg = item.get("knowledge_graph", item)
                        pred_docs[dk] = kg
            elif "results" in raw and isinstance(raw["results"], dict):
                pred_docs = raw["results"]
            else:
                # Fall back: apply same predictions to first gold doc
                pred_docs = {gold_docs[0]["doc_key"]: pred_kg}

    print(f"  {len(pred_docs)} document predictions loaded")

    # --- Evaluate ---
    metrics = evaluate_corpus(gold_docs, pred_docs, args.fuzzy_threshold)
    print_report(metrics)

    # --- Optional JSON output ---
    if args.output_json:
        # Convert PRF objects to dicts for serialization
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, PRF):
                serializable[k] = {
                    "precision": v.precision(),
                    "recall": v.recall(),
                    "f1": v.f1(),
                    "tp": v.tp,
                    "fp": v.fp,
                    "fn": v.fn,
                }
            elif isinstance(v, dict):
                serializable[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, PRF):
                        serializable[k][kk] = {
                            "precision": vv.precision(),
                            "recall": vv.recall(),
                            "f1": vv.f1(),
                            "tp": vv.tp,
                            "fp": vv.fp,
                            "fn": vv.fn,
                        }
                    else:
                        serializable[k][kk] = vv
            else:
                serializable[k] = v

        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
        print(f"\nMetrics written to {args.output_json}")


if __name__ == "__main__":
    main()
