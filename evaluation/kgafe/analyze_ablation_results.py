"""
Analyze ablation outputs and surface concrete failure patterns.

Example:
    python -m evaluation.kgafe.analyze_ablation_results \
        --input evaluation/results/ablation_smoke.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def fact_summary(result: Dict[str, Any]) -> Dict[str, int]:
    verdicts = result["metrics"]["verdict_distribution"]
    return {
        "supported": verdicts.get("supported", 0),
        "partially_supported": verdicts.get("partially_supported", 0),
        "contradicted": verdicts.get("contradicted", 0),
        "unverifiable": verdicts.get("unverifiable", 0),
        "total": verdicts.get("total", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KGAFE ablation results")
    parser.add_argument("--input", required=True, help="Path to ablation results JSON")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    runs: Dict[str, Dict[str, Any]] = data["runs"]
    run_names = list(runs.keys())
    print("Runs:", ", ".join(run_names))
    print()

    for name in run_names:
        agg = runs[name]["aggregate_metrics"]
        print(f"[{name}]")
        for field in [
            "avg_kgafe_score",
            "avg_kg_faithfulness",
            "avg_kg_precision",
            "avg_hallucination_rate",
            "avg_coverage",
            "num_questions",
            "total_facts_evaluated",
            "total_supported",
            "total_unverifiable",
        ]:
            print(f"  {field}: {agg.get(field)}")
        print()

    baseline = run_names[0]
    baseline_results = {
        item["question_id"]: item for item in runs[baseline]["individual_results"]
    }
    for challenger in run_names[1:]:
        print(f"=== {challenger} vs {baseline} ===")
        deltas: List[Dict[str, Any]] = []
        for item in runs[challenger]["individual_results"]:
            base = baseline_results.get(item["question_id"])
            if not base:
                continue
            delta = round(
                item["metrics"]["kgafe_score"] - base["metrics"]["kgafe_score"], 4
            )
            if delta == 0:
                continue
            deltas.append(
                {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type"),
                    "delta": delta,
                    "baseline": fact_summary(base),
                    "challenger": fact_summary(item),
                    "question": item["question"],
                }
            )

        deltas.sort(key=lambda x: x["delta"])
        print("Worst regressions:")
        for row in deltas[:5]:
            print(
                f"  {row['question_id']} [{row['question_type']}] delta={row['delta']}: "
                f"{row['question']}"
            )
            print(
                f"    baseline={row['baseline']} challenger={row['challenger']}"
            )
        print("Best improvements:")
        for row in list(reversed(deltas[-5:])):
            print(
                f"  {row['question_id']} [{row['question_type']}] delta={row['delta']}: "
                f"{row['question']}"
            )
            print(
                f"    baseline={row['baseline']} challenger={row['challenger']}"
            )
        print()


if __name__ == "__main__":
    main()
