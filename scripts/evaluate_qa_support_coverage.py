"""Evaluate whether a created KG contains gold-answer support for QA examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.qa_support import aggregate_support_results, evaluate_question_support
from multi_agent_kg.core import load_kg
from multi_agent_kg.core.governance import OrgChart


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "questions" in data:
            return data["questions"]
        if "examples" in data:
            return data["examples"]
        if "data" in data:
            return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported QA example format: {path}")


def _get_question_id(example: Dict[str, Any], index: int) -> str:
    return str(
        example.get("question_id")
        or example.get("id")
        or example.get("_id")
        or example.get("qid")
        or f"q{index:04d}"
    )


def _get_question(example: Dict[str, Any]) -> str:
    return str(example.get("question") or example.get("query") or "")


def _get_answer(example: Dict[str, Any]) -> str:
    answer = example.get("answer")
    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("answer")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return str(answer or example.get("gold_answer") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", required=True)
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--org-chart")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kg = load_kg(args.kg_path)
    org_chart = None
    if args.org_chart:
        data = json.loads(Path(args.org_chart).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("org_chart"), dict):
            data = data["org_chart"]
        org_chart = OrgChart.from_dict(data, kg)

    examples = _load_examples(Path(args.examples_json))
    results = []
    for index, example in enumerate(examples, start=1):
        question = _get_question(example)
        answer = _get_answer(example)
        if not question or not answer:
            continue
        result = evaluate_question_support(
            kg=kg,
            org_chart=org_chart,
            question_id=_get_question_id(example, index),
            question=question,
            gold_answer=answer,
        )
        results.append(result)

    payload = {
        "examples_json": args.examples_json,
        "kg_path": args.kg_path,
        "org_chart": args.org_chart,
        "aggregate": aggregate_support_results(results),
        "results": [result.to_dict() for result in results],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
