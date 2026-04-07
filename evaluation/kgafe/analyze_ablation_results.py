"""
Analyze ablation outputs and surface concrete failure patterns.

Example:
    python -m evaluation.kgafe.analyze_ablation_results \
        --input evaluation/results/ablation_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from statistics import mean
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


def paired_deltas(
    baseline_results: Dict[str, Dict[str, Any]],
    challenger_results: List[Dict[str, Any]],
    metric_name: str,
) -> List[float]:
    deltas: List[float] = []
    for item in challenger_results:
        base = baseline_results.get(item["question_id"])
        if not base:
            continue
        deltas.append(item["metrics"][metric_name] - base["metrics"][metric_name])
    return deltas


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 7) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(mean(draw))
    samples.sort()
    lo_idx = int(0.025 * (len(samples) - 1))
    hi_idx = int(0.975 * (len(samples) - 1))
    return {
        "mean": mean(values),
        "lo": samples[lo_idx],
        "hi": samples[hi_idx],
    }


def sign_test_pvalue(values: List[float]) -> float:
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    smaller = min(wins, losses)
    cumulative = 0.0
    for k in range(smaller + 1):
        cumulative += math.comb(n, k)
    p_one_tail = cumulative / (2 ** n)
    return min(1.0, 2 * p_one_tail)


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
        score_deltas = paired_deltas(
            baseline_results,
            runs[challenger]["individual_results"],
            "kgafe_score",
        )
        wins = sum(1 for delta in score_deltas if delta > 0)
        losses = sum(1 for delta in score_deltas if delta < 0)
        ties = sum(1 for delta in score_deltas if delta == 0)
        ci = bootstrap_ci(score_deltas)
        p_value = sign_test_pvalue(score_deltas)
        print(
            "  paired delta (kgafe): "
            f"{ci['mean']:.4f} [95% CI {ci['lo']:.4f}, {ci['hi']:.4f}]"
        )
        print(f"  win/loss/tie: {wins}/{losses}/{ties}")
        print(f"  sign-test p-value: {p_value:.4f}")

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
