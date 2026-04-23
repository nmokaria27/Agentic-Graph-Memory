#!/usr/bin/env python3
"""
Analyze an accumulated KG build against SciERC gold.

This is a lightweight diagnostics script for governed or unguided corpus builds.
It answers the questions we keep asking during iteration:

- Which relations are matching gold and which are broken?
- How many docs still have entities but zero triples?
- How many entities are orphaned (never used in any triple)?
- What does the graph density look like per document?

The script accepts a single accumulated KG JSON (governed or plain) and
compares it to the corresponding SciERC split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from evaluation.adapters.scierc_adapter import SciERCAdapter
from evaluation.evaluate_kg import evaluate_entities_strict, evaluate_triples_strict


def _load_accumulated(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = json.loads(Path(path).read_text())
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        return kg.get("entities", []), kg.get("triples", [])
    return data.get("entities", []), data.get("triples", [])


def _group_by_doc(
    entities: List[Dict[str, Any]],
    triples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"entities": [], "triples": []}
    )
    entity_lookup = {entity.get("id"): entity for entity in entities}

    for entity in entities:
        doc_id = entity.get("metadata", {}).get("source_document")
        if doc_id:
            grouped[doc_id]["entities"].append(entity)

    for triple in triples:
        doc_id = triple.get("source") or triple.get("metadata", {}).get("source_document")
        if doc_id:
            grouped[doc_id]["triples"].append(triple)
            for ent_id in [triple.get("subject"), triple.get("object")]:
                ent = entity_lookup.get(ent_id)
                if ent and ent not in grouped[doc_id]["entities"]:
                    grouped[doc_id]["entities"].append(ent)

    return grouped


def _relation_counter(triples: List[Dict[str, Any]]) -> Counter:
    return Counter(triple.get("relation", "") for triple in triples if triple.get("relation"))


def _orphan_stats(
    entities: List[Dict[str, Any]],
    triples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    used = set()
    for triple in triples:
        if triple.get("subject"):
            used.add(triple["subject"])
        if triple.get("object"):
            used.add(triple["object"])
    total = len(entities)
    orphan_ids = [entity.get("id") for entity in entities if entity.get("id") not in used]
    return {
        "total_entities": total,
        "orphan_entities": len(orphan_ids),
        "orphan_fraction": round(len(orphan_ids) / total, 4) if total else 0.0,
        "sample_orphans": orphan_ids[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an accumulated SciERC KG build.")
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=os.path.join("evaluation", "datasets", "scierc"))
    parser.add_argument("--max-docs", type=int, default=10)
    args = parser.parse_args()

    adapter = SciERCAdapter(os.path.join(args.data_dir, f"{args.split}.json"), skip_generic=True)
    gold_docs = adapter.documents[: args.max_docs]
    gold_by_doc = {
        doc["doc_key"]: {
            "entities": adapter.get_gold_entities(doc),
            "triples": adapter.get_gold_triples(doc),
        }
        for doc in gold_docs
    }

    entities, triples = _load_accumulated(args.kg_path)
    grouped = _group_by_doc(entities, triples)

    entity_prf_sum = {"tp": 0, "fp": 0, "fn": 0}
    triple_prf_sum = {"tp": 0, "fp": 0, "fn": 0}
    pred_rel_counts = Counter()
    gold_rel_counts = Counter()
    rel_tp = Counter()
    docs_with_zero_triples: List[str] = []
    per_doc_rows: List[Dict[str, Any]] = []

    for doc_id, gold in gold_by_doc.items():
        pred = grouped.get(doc_id, {"entities": [], "triples": []})
        ent_prf, _, _, _ = evaluate_entities_strict(gold["entities"], pred["entities"], map_types=True)
        tri_prf, per_rel, _ = evaluate_triples_strict(gold["triples"], pred["triples"])
        entity_prf_sum["tp"] += ent_prf.tp
        entity_prf_sum["fp"] += ent_prf.fp
        entity_prf_sum["fn"] += ent_prf.fn
        triple_prf_sum["tp"] += tri_prf.tp
        triple_prf_sum["fp"] += tri_prf.fp
        triple_prf_sum["fn"] += tri_prf.fn
        pred_rel_counts.update(_relation_counter(pred["triples"]))
        gold_rel_counts.update(_relation_counter(gold["triples"]))
        for rel, stats in per_rel.items():
            rel_tp[rel] += stats.tp
        if pred["entities"] and not pred["triples"]:
            docs_with_zero_triples.append(doc_id)
        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "pred_entities": len(pred["entities"]),
                "pred_triples": len(pred["triples"]),
                "gold_entities": len(gold["entities"]),
                "gold_triples": len(gold["triples"]),
                "triple_tp": tri_prf.tp,
            }
        )

    summary = {
        "kg_path": args.kg_path,
        "split": args.split,
        "max_docs": args.max_docs,
        "overall_counts": {
            "entities": len(entities),
            "triples": len(triples),
            **_orphan_stats(entities, triples),
        },
        "docs_with_entities_but_zero_triples": docs_with_zero_triples,
        "per_relation": [],
        "per_doc": per_doc_rows,
        "aggregate_prf_counts": {
            "entity": entity_prf_sum,
            "triple": triple_prf_sum,
        },
    }

    all_rels = sorted(set(pred_rel_counts) | set(gold_rel_counts) | set(rel_tp))
    for rel in all_rels:
        summary["per_relation"].append(
            {
                "relation": rel,
                "gold": gold_rel_counts.get(rel, 0),
                "pred": pred_rel_counts.get(rel, 0),
                "tp": rel_tp.get(rel, 0),
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
