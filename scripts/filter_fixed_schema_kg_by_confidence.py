#!/usr/bin/env python3
"""Create a confidence-filtered copy of a KG artifact.

Useful for fixed-schema benchmark runs where the extractor stores calibrated
confidence and we want the admitted/evaluated graph to exclude weak candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _confidence(triple: Dict[str, Any]) -> float:
    try:
        return float(triple.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter KG triples by confidence.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    kg = data.get("knowledge_graph", data)
    triples: List[Dict[str, Any]] = kg.get("triples", [])
    kept = [triple for triple in triples if _confidence(triple) >= args.min_confidence]
    kg["triples"] = kept

    if "stats" in data and isinstance(data["stats"], dict):
        data["stats"]["triples"] = len(kept)
        data["stats"]["confidence_filter_min"] = args.min_confidence
        data["stats"]["confidence_filter_removed"] = len(triples) - len(kept)

    Path(args.output).write_text(json.dumps(data, indent=2))
    print(
        f"wrote {args.output}: kept {len(kept)}/{len(triples)} triples "
        f"(min_confidence={args.min_confidence})"
    )


if __name__ == "__main__":
    main()
