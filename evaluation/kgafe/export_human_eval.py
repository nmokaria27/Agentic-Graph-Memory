"""
Export blinded human-evaluation sheets from an ablation result JSON.

Example:
    python -m evaluation.kgafe.export_human_eval \
        --input evaluation/results/scierc_compositional12_postfix_v2.json \
        --configs flat_basic domain_basic domain_advanced \
        --sample-size 12 \
        --out-prefix evaluation/results/human_eval_compositional12
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def build_question_rows(
    data: Dict[str, Any],
    configs: List[str],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    run_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for config in configs:
        run = data["runs"].get(config)
        if not run:
            raise ValueError(f"Config '{config}' not found in ablation results")
        run_maps[config] = {
            item["question_id"]: item for item in run["individual_results"]
        }

    shared_question_ids = sorted(
        set.intersection(*(set(results.keys()) for results in run_maps.values()))
    )
    if not shared_question_ids:
        raise ValueError("No shared question IDs across the requested configs")

    rng = random.Random(seed)
    rng.shuffle(shared_question_ids)
    selected_ids = shared_question_ids[: min(sample_size, len(shared_question_ids))]

    rows: List[Dict[str, Any]] = []
    for question_id in selected_ids:
        question = run_maps[configs[0]][question_id]
        shuffled = configs[:]
        rng.shuffle(shuffled)
        labels = [chr(ord("A") + idx) for idx in range(len(shuffled))]

        row: Dict[str, Any] = {
            "question_id": question_id,
            "question_type": question.get("question_type", ""),
            "difficulty": question.get("difficulty", ""),
            "question": question["question"],
            "gold_answer": question.get("gold_answer", ""),
        }
        answer_key = {}
        for label, config in zip(labels, shuffled):
            row[f"answer_{label}"] = run_maps[config][question_id]["answer"]
            row[f"correctness_{label}"] = ""
            row[f"usefulness_{label}"] = ""
            row[f"attribution_{label}"] = ""
            row[f"verbosity_{label}"] = ""
            answer_key[label] = config

        row["_answer_key"] = answer_key
        rows.append(row)

    return rows


def write_outputs(rows: List[Dict[str, Any]], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    key_path = out_prefix.with_name(out_prefix.name + "_key.json")

    if not rows:
        raise ValueError("No human-eval rows to write")

    fieldnames = [key for key in rows[0].keys() if key != "_answer_key"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "_answer_key"})

    key_payload = {
        row["question_id"]: row["_answer_key"]
        for row in rows
    }
    key_path.write_text(json.dumps(key_payload, indent=2), encoding="utf-8")

    print(f"Wrote blinded sheet: {csv_path}")
    print(f"Wrote answer key:    {key_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blinded human-evaluation sheets")
    parser.add_argument("--input", required=True, help="Ablation result JSON")
    parser.add_argument("--configs", nargs="+", required=True, help="Configs to compare")
    parser.add_argument("--sample-size", type=int, default=12, help="Number of shared questions")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    rows = build_question_rows(
        data=data,
        configs=args.configs,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    write_outputs(rows, Path(args.out_prefix))


if __name__ == "__main__":
    main()
