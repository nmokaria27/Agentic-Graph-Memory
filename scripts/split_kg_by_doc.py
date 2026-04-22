#!/usr/bin/env python3
"""
Split an accumulating governed KG artifact into per-document JSONs compatible
with evaluation/run_evaluation.py --skip-pipeline.

Entities are partitioned by metadata.source_document; triples by their top-level
source field. A doc is considered to own an entity if its id appears in that
document's triples or matches source_document. Per-doc files are written as
{"knowledge_graph": {"entities": [...], "triples": [...]}}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to governed KG JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for per-doc JSONs")
    parser.add_argument("--doc-ids", nargs="*", help="If provided, only emit these doc ids")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    kg = data.get("knowledge_graph", data)
    entities: List[Dict[str, Any]] = kg.get("entities", [])
    triples: List[Dict[str, Any]] = kg.get("triples", [])

    triples_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in triples:
        src = (t.get("source") or "").strip()
        if not src:
            continue
        triples_by_doc[src].append(t)

    entities_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entity_by_id: Dict[str, Dict[str, Any]] = {e["id"]: e for e in entities if e.get("id")}
    for e in entities:
        src = (e.get("metadata") or {}).get("source_document") or ""
        if not src:
            continue
        entities_by_doc[src].append(e)

    for doc_id, doc_triples in triples_by_doc.items():
        referenced: Set[str] = set()
        for t in doc_triples:
            if t.get("subject"):
                referenced.add(t["subject"])
            if t.get("object"):
                referenced.add(t["object"])
        existing_ids = {e["id"] for e in entities_by_doc[doc_id]}
        for eid in referenced:
            if eid in existing_ids:
                continue
            ent = entity_by_id.get(eid)
            if ent:
                entities_by_doc[doc_id].append(ent)

    doc_ids = set(entities_by_doc.keys()) | set(triples_by_doc.keys())
    if args.doc_ids:
        doc_ids &= set(args.doc_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    written = 0
    for doc_id in sorted(doc_ids):
        payload = {
            "knowledge_graph": {
                "entities": entities_by_doc.get(doc_id, []),
                "triples": triples_by_doc.get(doc_id, []),
            }
        }
        out_path = os.path.join(args.output_dir, f"{doc_id}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        written += 1
        print(
            f"  {doc_id}: {len(payload['knowledge_graph']['entities'])} entities, "
            f"{len(payload['knowledge_graph']['triples'])} triples"
        )

    print(f"\nWrote {written} per-doc files to {args.output_dir}")


if __name__ == "__main__":
    main()
