"""Score KG QA systems on prepared HotpotQA questions with EM/token-F1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import (
    aggregate_answer_metrics,
    answer_metrics,
    clean_prediction_for_scoring,
    load_prepared_examples,
)
from evaluation.kgafe.run_ablations import EXPERIMENTS, build_qa_system
from multi_agent_kg.core import LLMConfig, load_kg


def _load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("runs", {})


def _extract_answer(result: Dict[str, Any]) -> str:
    answer = result.get("final_answer") or result.get("answer") or ""
    if isinstance(answer, dict):
        answer = answer.get("answer", "")
    return str(answer).strip()


def _write_output(
    output: Path,
    *,
    args: argparse.Namespace,
    runs: Dict[str, Dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": "HotpotQA distractor dev pilot",
                "kg_path": args.kg_path,
                "org_chart": args.org_chart,
                "model": args.model,
                "examples_json": args.examples_json,
                "runs": runs,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--org-chart")
    parser.add_argument("--examples-json", default="evaluation/results/hotpotqa_pilot_20.json")
    parser.add_argument("--configs", nargs="+", default=["flat_path_basic", "graphrag_basic", "domain_basic", "domain_advanced"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--output", default="evaluation/results/hotpotqa_kg_qa_20.json")
    args = parser.parse_args()

    examples = load_prepared_examples(Path(args.examples_json))
    kg = load_kg(args.kg_path)
    llm_config = LLMConfig(model=args.model)
    output = Path(args.output)
    runs = _load_existing(output)

    for config_name in args.configs:
        config = EXPERIMENTS[config_name]
        rows: List[Dict[str, Any]] = runs.get(config_name, {}).get("results", [])
        completed = {row.get("question_id") for row in rows}
        qa_system = build_qa_system(
            kg=kg,
            llm_config=llm_config,
            config=config,
            org_chart_path=args.org_chart,
        )
        for i, example in enumerate(examples, start=1):
            if example.question_id in completed:
                print(f"[{config_name}] skip {i}/{len(examples)} {example.question_id}", flush=True)
                continue
            print(f"[{config_name}] {i}/{len(examples)} {example.question_id}", flush=True)
            start = time.time()
            result = qa_system.query(example.question)
            answer = _extract_answer(result)
            row = {
                "question_id": example.question_id,
                "question": example.question,
                "gold_answer": example.answer,
                "answer": answer,
                "scored_answer": clean_prediction_for_scoring(answer, example.answer),
                "metrics": answer_metrics(answer, example.answer),
                "duration_seconds": round(time.time() - start, 3),
                "system_metadata": result,
            }
            rows.append(row)
            runs[config_name] = {
                "config": asdict(config),
                "aggregate_metrics": aggregate_answer_metrics(rows),
                "results": rows,
            }
            _write_output(output, args=args, runs=runs)
            print(
                f"[{config_name}] answer={answer[:100]!r} gold={example.answer!r} "
                f"em={row['metrics']['exact_match']:.1f} f1={row['metrics']['token_f1']:.3f}",
                flush=True,
            )
    _write_output(output, args=args, runs=runs)
    print(json.dumps({name: run["aggregate_metrics"] for name, run in runs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
