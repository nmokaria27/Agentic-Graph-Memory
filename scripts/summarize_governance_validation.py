#!/usr/bin/env python3
"""Summarize governance validation artifacts into compact table rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _kg_stats(path: Optional[str]) -> Dict[str, Any]:
    data = _load(path)
    if not data:
        return {}
    kg = data.get("knowledge_graph", data)
    triples = kg.get("triples", [])
    supported = 0
    for triple in triples:
        metadata = triple.get("metadata") if isinstance(triple.get("metadata"), dict) else {}
        if metadata.get("source_supported") or metadata.get("evidence"):
            supported += 1
    return {
        "triple_count": len(triples),
        "source_supported_triples": supported,
        "source_supported_fraction": supported / len(triples) if triples else 0.0,
    }


def _metric(score: Dict[str, Any], path: str, default: float = 0.0) -> float:
    value: Any = score
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return float(value)


def _row(name: str, kg_path: str, score_path: str, stats_path: str = "") -> Dict[str, Any]:
    score = _load(score_path)
    stats = _load(stats_path)
    kg = _kg_stats(kg_path)
    return {
        "condition": name,
        "triple_f1_strict": _metric(score, "triple_strict.f1"),
        "triple_f1_mapped_fuzzy": _metric(score, "triple_mapped.f1"),
        "triple_precision_strict": _metric(score, "triple_strict.precision"),
        "triple_recall_strict": _metric(score, "triple_strict.recall"),
        "triple_hallucination_rate": _metric(score, "triple_hallucination_rate"),
        "triple_count": kg.get("triple_count", stats.get("triples", 0)),
        "zero_triple_docs": stats.get("zero_triple_docs", ""),
        "source_supported_triples": kg.get("source_supported_triples", ""),
        "source_supported_fraction": kg.get("source_supported_fraction", ""),
        "decision_counts": stats.get("decision_counts", {}),
        "kg_path": kg_path,
        "score_path": score_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        action="append",
        nargs="+",
        metavar="FIELD",
        help="Row as: NAME KG_PATH SCORE_PATH [STATS_PATH]. Repeatable.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for spec in args.row or []:
        if len(spec) < 3:
            raise SystemExit("--row requires at least NAME KG SCORE")
        name, kg_path, score_path = spec[:3]
        stats_path = spec[3] if len(spec) > 3 else ""
        rows.append(_row(name, kg_path, score_path, stats_path))

    payload = {"rows": rows}
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
