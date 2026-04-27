#!/usr/bin/env python3
"""Compare a governed (triage) and an ungoverned SciERC 50-doc build.

Reads existing governed artifact + new ungoverned artifact (plus F1 + analyze
outputs on both sides) and emits every metric requested for the paper's main
50-doc comparison:

- entity strict F1 (both sides)
- triple strict F1 (both sides)
- entity hallucination rate (both sides)
- triple hallucination rate (both sides)
- entity overlap (Jaccard)
- triple overlap (Jaccard)
- cross-domain fraction (governed)
- governance overhead seconds + pct
- orphan fraction delta
- zero-triple-doc delta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def _load(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _entities(kg_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    kg = kg_json.get("knowledge_graph", kg_json)
    ents = kg.get("entities", [])
    if isinstance(ents, dict):
        ents = list(ents.values())
    return ents


def _triples(kg_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    kg = kg_json.get("knowledge_graph", kg_json)
    return list(kg.get("triples", []))


def _entity_surface(e: Dict[str, Any]) -> str:
    if e.get("text"):
        return str(e["text"]).strip().lower()
    labels = e.get("labels") or []
    if labels:
        return str(labels[0]).strip().lower()
    return str(e.get("id", "")).strip().lower()


def _entity_set(entities: Sequence[Dict[str, Any]]) -> Set[str]:
    return {s for e in entities if (s := _entity_surface(e))}


def _triple_surface(t: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Tuple[str, str, str]:
    meta = t.get("metadata", {}) or {}
    subj = meta.get("original_subject")
    obj = meta.get("original_object")
    if not subj:
        se = lookup.get(t.get("subject", ""))
        subj = _entity_surface(se) if se else str(t.get("subject", "")).strip().lower()
    if not obj:
        oe = lookup.get(t.get("object", ""))
        obj = _entity_surface(oe) if oe else str(t.get("object", "")).strip().lower()
    return (str(subj).strip().lower(), str(t.get("relation", "")).strip(), str(obj).strip().lower())


def _triple_set(entities: Sequence[Dict[str, Any]], triples: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    lookup = {e.get("id"): e for e in entities}
    return {_triple_surface(t, lookup) for t in triples if t.get("relation")}


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return round(len(a & b) / u, 4) if u else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--governed-kg", default="evaluation/results/governed_created_50docs_triage.json")
    p.add_argument("--ungoverned-kg", default="evaluation/results/ungoverned_created_50docs_v3.json")
    p.add_argument("--governed-f1", default="evaluation/results/f1_governed_50docs_triage.json")
    p.add_argument("--ungoverned-f1", default="evaluation/results/f1_ungoverned_50docs_v3.json")
    p.add_argument("--governed-analyze", default="evaluation/results/analyze_governed_50docs_triage.json")
    p.add_argument("--ungoverned-analyze", default="evaluation/results/analyze_ungoverned_50docs_v3.json")
    p.add_argument(
        "--governed-elapsed-seconds",
        type=float,
        default=43396.8,
        help="Governed processing time (sum of per-doc Time markers from /tmp/run_triage_50doc.log)",
    )
    p.add_argument(
        "--ungoverned-stats",
        default="evaluation/results/ungoverned_50docs_v3_stats.json",
        help="Ungoverned run stats JSON (for elapsed_seconds)",
    )
    p.add_argument("--output", default="evaluation/results/compare_50docs_v3.json")
    args = p.parse_args()

    g_kg = _load(args.governed_kg)
    u_kg = _load(args.ungoverned_kg)
    g_f1 = _load(args.governed_f1)
    u_f1 = _load(args.ungoverned_f1)
    g_an = _load(args.governed_analyze)
    u_an = _load(args.ungoverned_analyze)

    try:
        u_stats = _load(args.ungoverned_stats)
        ungoverned_elapsed = float(u_stats.get("elapsed_seconds", 0.0))
    except FileNotFoundError:
        ungoverned_elapsed = 0.0

    g_ents = _entities(g_kg)
    u_ents = _entities(u_kg)
    g_trips = _triples(g_kg)
    u_trips = _triples(u_kg)

    g_eset = _entity_set(g_ents)
    u_eset = _entity_set(u_ents)
    g_tset = _triple_set(g_ents, g_trips)
    u_tset = _triple_set(u_ents, u_trips)

    triage = g_kg.get("triage_stats", {}) or {}
    total_triples = len(g_trips) or 1
    cross_domain = (triage.get("review_reasons", {}) or {}).get("cross_domain", 0)
    cross_domain_fraction = round(cross_domain / total_triples, 4)

    overhead_seconds = round(args.governed_elapsed_seconds - ungoverned_elapsed, 2) if ungoverned_elapsed else None
    overhead_pct = round((overhead_seconds / ungoverned_elapsed) * 100, 2) if ungoverned_elapsed else None

    def _f1_block(f1: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_strict_f1": f1.get("entity_strict", {}).get("f1"),
            "triple_strict_f1": f1.get("triple_strict", {}).get("f1"),
            "entity_hallucination_rate": f1.get("entity_hallucination_rate"),
            "triple_hallucination_rate": f1.get("triple_hallucination_rate"),
        }

    def _analyze_block(an: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entities": an.get("entities"),
            "triples": an.get("triples"),
            "orphan_fraction": an.get("orphan_fraction"),
            "docs_with_zero_triples": an.get("docs_with_zero_triples"),
            "docs_with_entities_no_triples": an.get("docs_with_entities_no_triples"),
        }

    comparison = {
        "governed": {
            **_f1_block(g_f1),
            **_analyze_block(g_an),
            "kg_entities": len(g_ents),
            "kg_triples": len(g_trips),
            "triage_stats": triage,
            "elapsed_seconds": args.governed_elapsed_seconds,
        },
        "ungoverned": {
            **_f1_block(u_f1),
            **_analyze_block(u_an),
            "kg_entities": len(u_ents),
            "kg_triples": len(u_trips),
            "elapsed_seconds": ungoverned_elapsed,
        },
        "comparison": {
            "entity_overlap_jaccard": _jaccard(g_eset, u_eset),
            "triple_overlap_jaccard": _jaccard(g_tset, u_tset),
            "unique_to_governed_triples": len(g_tset - u_tset),
            "unique_to_ungoverned_triples": len(u_tset - g_tset),
            "cross_domain_fraction": cross_domain_fraction,
            "governance_overhead_seconds": overhead_seconds,
            "governance_overhead_pct": overhead_pct,
            "orphan_fraction_delta": (
                round((g_an.get("orphan_fraction") or 0) - (u_an.get("orphan_fraction") or 0), 4)
                if g_an.get("orphan_fraction") is not None and u_an.get("orphan_fraction") is not None
                else None
            ),
            "zero_triple_docs_delta": (
                (g_an.get("docs_with_zero_triples") or 0) - (u_an.get("docs_with_zero_triples") or 0)
                if g_an.get("docs_with_zero_triples") is not None and u_an.get("docs_with_zero_triples") is not None
                else None
            ),
        },
    }

    Path(args.output).write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(json.dumps(comparison, indent=2, default=str))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
