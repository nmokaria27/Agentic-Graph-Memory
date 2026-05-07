#!/usr/bin/env python3
"""Canonicalize SciERC relation labels in a KG artifact.

GPT-5 sometimes emits relation labels as ``used_for`` while SciERC gold uses
``Used-for``. This utility keeps the graph unchanged except for schema-label
normalization so strict scoring measures extraction quality rather than casing
or separator artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALIASES = {
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


def canonical_relation(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.upper().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    kg = payload.get("knowledge_graph", payload)
    changed = 0
    for triple in kg.get("triples", []):
        old = triple.get("relation")
        new = canonical_relation(old)
        if new != old:
            triple["relation"] = new
            changed += 1
    payload.setdefault("postprocessing", {})["scierc_relation_canonicalized"] = {
        "changed_triples": changed,
        "input": args.input,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"input": args.input, "output": args.output, "changed_triples": changed}, indent=2))


if __name__ == "__main__":
    main()
