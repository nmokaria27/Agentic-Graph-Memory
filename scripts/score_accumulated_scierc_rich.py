#!/usr/bin/env python3
"""Score an accumulated SciERC KG with strict and semantic-normalized metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from evaluation.adapters.scierc_adapter import SciERCAdapter
from evaluation.evaluate_kg import evaluate_corpus, PRF


def _load_kg(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    return data.get("knowledge_graph", data)


def _group_by_doc(kg: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    entities = kg.get("entities", [])
    triples = kg.get("triples", [])
    entity_by_id = {entity.get("id"): entity for entity in entities}

    for entity in entities:
        doc_id = entity.get("metadata", {}).get("source_document")
        if not doc_id:
            continue
        grouped.setdefault(doc_id, {"entities": [], "triples": []})["entities"].append(entity)

    for triple in triples:
        doc_id = triple.get("source") or triple.get("metadata", {}).get("source_document")
        if not doc_id:
            continue
        bucket = grouped.setdefault(doc_id, {"entities": [], "triples": []})
        bucket["triples"].append(triple)
        for entity_id in (triple.get("subject"), triple.get("object")):
            entity = entity_by_id.get(entity_id)
            if entity and entity not in bucket["entities"]:
                bucket["entities"].append(entity)

    return grouped


def _serialize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, PRF):
            result[key] = {
                "precision": value.precision(),
                "recall": value.recall(),
                "f1": value.f1(),
                "tp": value.tp,
                "fp": value.fp,
                "fn": value.fn,
            }
        elif isinstance(value, dict):
            result[key] = _serialize(value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=os.path.join("evaluation", "datasets", "scierc"))
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = SciERCAdapter(os.path.join(args.data_dir, f"{args.split}.json"), skip_generic=True)
    gold_data = adapter.get_all_gold(max_docs=args.max_docs)
    grouped = _group_by_doc(_load_kg(args.kg_path))
    doc_keys = {doc["doc_key"] for doc in gold_data}
    pred_docs = {doc_id: grouped.get(doc_id, {"entities": [], "triples": []}) for doc_id in doc_keys}
    metrics = evaluate_corpus(gold_data, pred_docs, fuzzy_threshold=args.fuzzy_threshold)
    serialized = _serialize(metrics)
    serialized["kg_path"] = args.kg_path
    serialized["max_docs"] = args.max_docs
    serialized["fuzzy_threshold"] = args.fuzzy_threshold
    Path(args.output).write_text(json.dumps(serialized, indent=2))
    print(json.dumps(serialized, indent=2))


if __name__ == "__main__":
    main()
