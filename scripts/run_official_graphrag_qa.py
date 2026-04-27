"""Run official Microsoft GraphRAG on an existing KGAFE question set.

This script treats GraphRAG as an external corpus-RAG system: GraphRAG indexes
the source documents under --graphrag-root, answers the same saved benchmark
questions, and KGAFE scores those answers against the governed KG gold facts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.kgafe.benchmark_generator import BenchmarkQuestion
from evaluation.kgafe.evaluator import BenchmarkResult, KGAFEEvaluator
from evaluation.kgafe.run_ablations import load_org_chart, summarize_comparison
from multi_agent_kg.core import load_kg
from multi_agent_kg.core.knowledge_graph import Triple


class OfficialGraphRAGQA:
    """Adapter exposing official GraphRAG CLI as a KGAFE-compatible QA system."""

    def __init__(
        self,
        *,
        root: Path,
        method: str,
        response_type: str,
        timeout_seconds: int,
    ) -> None:
        self.root = root
        self.method = method
        self.response_type = response_type
        self.timeout_seconds = timeout_seconds
        self.graphrag_bin = shutil.which("graphrag")

    def query(self, question: str) -> Dict[str, Any]:
        cmd = [
            self.graphrag_bin or "graphrag",
            "query",
            "--root",
            str(self.root),
            "--method",
            self.method,
            "--response-type",
            self.response_type,
            question,
        ]
        start = time.time()
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            return {
                "answer": "",
                "final_answer": "",
                "retrieval_mode": f"official_graphrag_{self.method}",
                "returncode": -1,
                "stderr": f"GraphRAG query timed out after {self.timeout_seconds}s",
                "raw_stdout": (exc.stdout or "")[-8000:],
                "duration_seconds": duration,
            }
        duration = time.time() - start
        answer = _clean_graphrag_stdout(completed.stdout)
        if completed.returncode != 0:
            answer = ""
        return {
            "answer": answer,
            "final_answer": answer,
            "retrieval_mode": f"official_graphrag_{self.method}",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
            "raw_stdout": completed.stdout[-8000:],
            "duration_seconds": duration,
        }


def _clean_graphrag_stdout(stdout: str) -> str:
    lines = [line.rstrip() for line in stdout.splitlines()]
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("INFO:") or stripped.startswith("SUCCESS:"):
            continue
        if stripped.startswith("WARNING:") or stripped.startswith("ERROR:"):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _load_questions(path: Path) -> List[BenchmarkQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion(**question) for question in data["questions"]]


def _write_payload(
    *,
    output: Path,
    args: argparse.Namespace,
    questions: List[BenchmarkQuestion],
    run_name: str,
    result: BenchmarkResult,
) -> None:
    aggregate = result.compute_aggregates()
    runs = {
        run_name: {
            "config": {
                "name": run_name,
                "description": "Official Microsoft GraphRAG over the original 50-document SciERC corpus",
                "method": args.method,
                "response_type": args.response_type,
            },
            "aggregate_metrics": aggregate,
            "individual_results": [item.to_dict() for item in result.individual_results],
        }
    }
    payload = {
        "kg_path": args.kg_path,
        "model": args.model,
        "judge_panel_enabled": False,
        "questions": [asdict(question) for question in questions],
        "runs": runs,
        "comparison_summary": summarize_comparison(runs),
        "completed_questions": len(result.individual_results),
        "total_questions": len(questions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--org-chart")
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--graphrag-root", required=True)
    parser.add_argument("--method", default="local", choices=["local", "global", "drift", "basic"])
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--response-type", default="Single concise answer")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kg = load_kg(args.kg_path)
    org_chart = load_org_chart(args.org_chart, kg) if args.org_chart else None
    evaluator = KGAFEEvaluator(
        kg=kg,
        model=args.model,
        enable_judge_panel=False,
        org_chart=org_chart,
    )
    questions = _load_questions(Path(args.questions_json))
    qa_system = OfficialGraphRAGQA(
        root=Path(args.graphrag_root),
        method=args.method,
        response_type=args.response_type,
        timeout_seconds=args.timeout_seconds,
    )

    run_name = f"official_graphrag_{args.method}"
    output = Path(args.output)
    benchmark = BenchmarkResult()

    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        prior = existing.get("runs", {}).get(run_name, {}).get("individual_results", [])
        benchmark.individual_results = []
        if prior:
            from evaluation.kgafe.evaluator import EvaluationResult, KGAFEMetrics

            for item in prior:
                metrics = KGAFEMetrics(**{
                    key: value
                    for key, value in item["metrics"].items()
                    if key in KGAFEMetrics.__dataclass_fields__
                })
                benchmark.individual_results.append(
                    EvaluationResult(
                        question=item["question"],
                        answer=item["answer"],
                        metrics=metrics,
                        question_id=item.get("question_id"),
                        question_type=item.get("question_type"),
                        difficulty=item.get("difficulty"),
                        atomic_facts=item.get("atomic_facts", []),
                        verification_results=item.get("verification_results", []),
                        judge_verdict=item.get("judge_verdict"),
                        gold_answer=item.get("gold_answer"),
                        aux_metrics=item.get("aux_metrics", {}),
                        system_metadata=item.get("system_metadata", {}),
                        duration_seconds=item.get("duration_seconds", 0.0),
                    )
                )

    completed_ids = {result.question_id for result in benchmark.individual_results}
    for i, question in enumerate(questions, start=1):
        if question.question_id in completed_ids:
            print(f"[{i}/{len(questions)}] skip {question.question_id}: already complete", flush=True)
            continue

        print(f"[{i}/{len(questions)}] official GraphRAG query: {question.question_id}", flush=True)
        qa_result = qa_system.query(question.question)
        answer = qa_result.get("final_answer", "")
        print(f"[{i}/{len(questions)}] answer chars={len(answer)} returncode={qa_result.get('returncode')}", flush=True)

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
        _write_payload(
            output=output,
            args=args,
            questions=questions,
            run_name=run_name,
            result=benchmark,
        )
        print(f"[{i}/{len(questions)}] checkpoint -> {output}", flush=True)

    _write_payload(output=output, args=args, questions=questions, run_name=run_name, result=benchmark)
    print(json.dumps(benchmark.compute_aggregates(), indent=2))


if __name__ == "__main__":
    main()
