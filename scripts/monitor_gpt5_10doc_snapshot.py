#!/usr/bin/env python3
"""Write and score a 10-doc GPT-5 SciERC snapshot once a 50-doc checkpoint reaches it."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.adapters.scierc_adapter import SciERCAdapter
CHECKPOINT = ROOT / "evaluation/results/scierc_governed_gpt5_test_50_schemahard.json.checkpoint.json"
SNAPSHOT = ROOT / "evaluation/results/scierc_governed_gpt5_test_10_schemahard.json"
ANALYSIS = ROOT / "evaluation/results/analyze_governed_gpt5_schemahard_10docs.json"
MIN_CONFIDENCE = float(os.getenv("FIXED_SCHEMA_MIN_TRIPLE_CONFIDENCE", "0.75"))


def main() -> None:
    adapter = SciERCAdapter(str(ROOT / "evaluation/datasets/scierc/test.json"), skip_generic=True)
    first10 = [doc["doc_key"] for doc in adapter.documents[:10]]

    while True:
        if not CHECKPOINT.exists():
            print(f"waiting_for_checkpoint={CHECKPOINT}", flush=True)
            time.sleep(60)
            continue
        data = json.loads(CHECKPOINT.read_text())
        processed = set(data.get("_processed_doc_ids", []))
        print(f"processed={len(processed)} missing={[doc for doc in first10 if doc not in processed]}", flush=True)

        if set(first10).issubset(processed):
            kg = data.get("knowledge_graph", {})
            filtered = copy.deepcopy(data)
            filtered_kg = filtered.setdefault("knowledge_graph", {})
            filtered_kg["entities"] = [
                entity
                for entity in kg.get("entities", [])
                if entity.get("metadata", {}).get("source_document") in first10
            ]
            allowed_entities = {entity.get("id") for entity in filtered_kg["entities"]}
            filtered_kg["triples"] = [
                triple
                for triple in kg.get("triples", [])
                if (
                    triple.get("source") in first10
                    or triple.get("metadata", {}).get("source_document") in first10
                )
                and triple.get("subject") in allowed_entities
                and triple.get("object") in allowed_entities
                and float(triple.get("confidence", 0.0) or 0.0) >= MIN_CONFIDENCE
            ]
            filtered["_processed_doc_ids"] = first10
            filtered.setdefault("stats", {})["entities"] = len(filtered_kg["entities"])
            filtered.setdefault("stats", {})["triples"] = len(filtered_kg["triples"])
            SNAPSHOT.write_text(json.dumps(filtered, indent=2))

            with ANALYSIS.open("w") as handle:
                subprocess.run(
                    [
                        str(ROOT / ".venv/bin/python"),
                        "scripts/analyze_scierc_build.py",
                        "--kg-path",
                        str(SNAPSHOT.relative_to(ROOT)),
                        "--split",
                        "test",
                        "--max-docs",
                        "10",
                    ],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            print(f"wrote={SNAPSHOT} analysis={ANALYSIS}", flush=True)
            return

        time.sleep(60)


if __name__ == "__main__":
    main()
