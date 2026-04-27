"""Run KGAFE QA configs on a previously generated question set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.kgafe.benchmark_generator import BenchmarkQuestion
from evaluation.kgafe.evaluator import BenchmarkResult, KGAFEEvaluator
from evaluation.kgafe.run_ablations import EXPERIMENTS, build_qa_system, load_org_chart, summarize_comparison
from multi_agent_kg.core import LLMConfig, load_kg
from multi_agent_kg.core.knowledge_graph import Triple


def _load_questions(path: Path) -> List[BenchmarkQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion(**question) for question in data["questions"]]


def _write_output(
    output: Path,
    *,
    kg_path: str,
    model: str,
    questions: List[BenchmarkQuestion],
    runs: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "kg_path": kg_path,
        "model": model,
        "judge_panel_enabled": False,
        "questions": [asdict(question) for question in questions],
        "runs": runs,
        "comparison_summary": summarize_comparison(runs),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", required=True, help="KG used by QA systems")
    parser.add_argument("--org-chart", help="Org chart used by QA systems")
    parser.add_argument("--eval-kg-path", help="Clean KG used by the evaluator; defaults to --kg-path")
    parser.add_argument("--eval-org-chart", help="Clean org chart used by the evaluator; defaults to --org-chart")
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--configs", nargs="+", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    qa_kg = load_kg(args.kg_path)
    org_chart = load_org_chart(args.org_chart, qa_kg) if args.org_chart else None
    eval_kg_path = args.eval_kg_path or args.kg_path
    eval_org_chart_path = args.eval_org_chart or args.org_chart
    eval_kg = load_kg(eval_kg_path)
    eval_org_chart = load_org_chart(eval_org_chart_path, eval_kg) if eval_org_chart_path else None
    evaluator = KGAFEEvaluator(
        kg=eval_kg,
        model=args.model,
        enable_judge_panel=False,
        org_chart=eval_org_chart,
    )
    questions = _load_questions(Path(args.questions_json))
    llm_config = LLMConfig(model=args.model)
    runs: Dict[str, Dict[str, Any]] = {}
    output = Path(args.output)

    for config_name in args.configs:
        config = EXPERIMENTS[config_name]
        print(f"=== {config_name} ===", flush=True)
        qa_system = build_qa_system(
            kg=qa_kg,
            llm_config=llm_config,
            config=config,
            org_chart_path=args.org_chart,
        )
        benchmark = BenchmarkResult()
        for i, question in enumerate(questions, start=1):
            print(f"[{config_name}] {i}/{len(questions)} {question.question_id}", flush=True)
            if hasattr(qa_system, "query_benchmark_question"):
                qa_result = qa_system.query_benchmark_question(question)
            else:
                qa_result = qa_system.query(question.question)
            answer = qa_result.get("final_answer", "")
            relevant = [
                Triple(subject=t["subject"], relation=t["relation"], object=t["object"])
                for t in question.supporting_triples
            ]
            eval_result = evaluator.evaluate_answer(
                question=question.question,
                answer=answer,
                gold_answer=question.gold_answer,
                relevant_triples=relevant,
            )
            eval_result.question_id = question.question_id
            eval_result.question_type = question.question_type
            eval_result.difficulty = question.difficulty
            eval_result.system_metadata = evaluator._extract_system_metadata(qa_result)
            eval_result.aux_metrics = evaluator._compute_aux_metrics(
                question=question,
                answer=answer,
                system_metadata=eval_result.system_metadata,
            )
            evaluator._apply_negative_abstention_credit(
                eval_result=eval_result,
                benchmark_question=question,
            )
            benchmark.individual_results.append(eval_result)
            runs[config_name] = {
                "config": asdict(config),
                "aggregate_metrics": benchmark.compute_aggregates(),
                "individual_results": [result.to_dict() for result in benchmark.individual_results],
            }
            _write_output(
                output,
                kg_path=args.kg_path,
                model=args.model,
                questions=questions,
                runs=runs,
            )
        print(json.dumps(runs[config_name]["aggregate_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
