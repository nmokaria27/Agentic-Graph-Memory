"""
Run controlled KGAFE ablations on a fixed benchmark set.

Example:
    python -m evaluation.kgafe.run_ablations \
        --kg-path kg_export.json \
        --n-questions 20 \
        --configs flat_basic domain_basic domain_advanced domain_no_debate \
        --no-judge \
        --output evaluation/results/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multi_agent_kg.core import LLMConfig, load_kg
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
from multi_agent_kg.core.domain_experts import DomainBuilder, OrgChart, QAOrchestrator
from evaluation.kgafe.baselines import build_baseline_system
from evaluation.kgafe.evaluator import KGAFEEvaluator


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    orchestrator: str  # basic | advanced
    num_domains: Optional[int] = None
    enable_debate: bool = True
    enable_critic: bool = True
    max_exploration_rounds: int = 3


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "rag_basic": ExperimentConfig(
        name="rag_basic",
        description="Document-RAG baseline over the source corpus",
        orchestrator="basic",
        num_domains=1,
    ),
    "flat_basic": ExperimentConfig(
        name="flat_basic",
        description="Single global expert over the full KG",
        orchestrator="basic",
        num_domains=1,
    ),
    "flat_path_basic": ExperimentConfig(
        name="flat_path_basic",
        description="Single global expert with path-focused retrieval",
        orchestrator="basic",
        num_domains=1,
    ),
    "graphrag_basic": ExperimentConfig(
        name="graphrag_basic",
        description="Graph-aware retrieval baseline using community summaries",
        orchestrator="basic",
        num_domains=1,
    ),
    "domain_basic": ExperimentConfig(
        name="domain_basic",
        description="Clustered domain experts without active exploration/debate",
        orchestrator="basic",
    ),
    "oracle_domain_basic": ExperimentConfig(
        name="oracle_domain_basic",
        description="Clustered domain experts with oracle routing to expected domains",
        orchestrator="basic",
    ),
    "domain_advanced": ExperimentConfig(
        name="domain_advanced",
        description="Clustered experts with exploration, debate, and critic",
        orchestrator="advanced",
        enable_debate=True,
        enable_critic=True,
        max_exploration_rounds=3,
    ),
    "domain_no_debate": ExperimentConfig(
        name="domain_no_debate",
        description="Advanced orchestrator without cross-domain debate",
        orchestrator="advanced",
        enable_debate=False,
        enable_critic=True,
        max_exploration_rounds=3,
    ),
    "domain_no_critic": ExperimentConfig(
        name="domain_no_critic",
        description="Advanced orchestrator without critic revision",
        orchestrator="advanced",
        enable_debate=True,
        enable_critic=False,
        max_exploration_rounds=3,
    ),
    "domain_passive": ExperimentConfig(
        name="domain_passive",
        description="Advanced orchestrator reduced to one exploration round and no debate/critic",
        orchestrator="advanced",
        enable_debate=False,
        enable_critic=False,
        max_exploration_rounds=1,
    ),
}


def load_org_chart(path: str, kg) -> OrgChart:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return OrgChart.from_dict(data, kg)


def build_qa_system(
    kg,
    llm_config: LLMConfig,
    config: ExperimentConfig,
    org_chart_path: Optional[str] = None,
):
    if org_chart_path and config.num_domains is None and os.path.exists(org_chart_path):
        org_chart = load_org_chart(org_chart_path, kg)
    else:
        builder = DomainBuilder(llm_config, target_num_domains=config.num_domains)
        org_chart = builder.build(kg)

    if config.name in {"rag_basic", "flat_path_basic", "graphrag_basic", "oracle_domain_basic"}:
        baseline = build_baseline_system(
            config.name,
            kg=kg,
            llm_config=llm_config,
            org_chart=org_chart,
            advanced_orchestrator_cls=AdvancedQAOrchestrator,
        )
        return baseline.qa_system

    if config.orchestrator == "advanced":
        return AdvancedQAOrchestrator(
            org_chart=org_chart,
            full_kg=kg,
            llm_config=llm_config,
            max_exploration_rounds=config.max_exploration_rounds,
            enable_debate=config.enable_debate,
            enable_critic=config.enable_critic,
        )

    return QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=llm_config)


def summarize_comparison(results_by_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = list(results_by_config.keys())
    if not names:
        return {}

    baseline = names[0]
    baseline_metrics = results_by_config[baseline]["aggregate_metrics"]
    summary: Dict[str, Any] = {"baseline": baseline, "comparisons": {}}

    for name in names[1:]:
        metrics = results_by_config[name]["aggregate_metrics"]
        summary["comparisons"][name] = {
            "delta_kgafe_score": round(
                metrics["avg_kgafe_score"] - baseline_metrics["avg_kgafe_score"], 4
            ),
            "delta_faithfulness": round(
                metrics["avg_kg_faithfulness"] - baseline_metrics["avg_kg_faithfulness"], 4
            ),
            "delta_precision": round(
                metrics["avg_kg_precision"] - baseline_metrics["avg_kg_precision"], 4
            ),
            "delta_hallucination_rate": round(
                metrics["avg_hallucination_rate"] - baseline_metrics["avg_hallucination_rate"], 4
            ),
            "delta_coverage": round(
                metrics["avg_coverage"] - baseline_metrics["avg_coverage"], 4
            ),
        }

    return summary


def write_output(
    output_path: str,
    args: argparse.Namespace,
    questions: List[Dict[str, Any]],
    runs: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "kg_path": args.kg_path,
        "model": args.model,
        "judge_panel_enabled": not args.no_judge,
        "question_types": args.question_types,
        "questions": questions,
        "runs": runs,
        "comparison_summary": summarize_comparison(runs),
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-question KGAFE ablations")
    parser.add_argument("--kg-path", required=True, help="Path to KG JSON")
    parser.add_argument("--org-chart", help="Optional cached org chart path for clustered variants")
    parser.add_argument("--n-questions", type=int, default=20, help="Number of benchmark questions")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["flat_basic", "domain_basic", "domain_passive", "domain_advanced"],
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment presets to run",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=["single_hop", "multi_hop", "aggregation", "comparison", "negative", "cross_domain"],
        help="Question types to include",
    )
    parser.add_argument("--model", default="gemma3:27b", help="Model name for the QA system/evaluator")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge panel")
    parser.add_argument("--output", required=True, help="Where to save ablation results JSON")
    args = parser.parse_args()

    if args.org_chart is None and os.path.exists("org_chart_cache.json"):
        args.org_chart = "org_chart_cache.json"

    kg = load_kg(args.kg_path)
    evaluator_org_chart = load_org_chart(args.org_chart, kg) if args.org_chart and os.path.exists(args.org_chart) else None
    evaluator = KGAFEEvaluator(
        kg=kg,
        model=args.model,
        enable_judge_panel=not args.no_judge,
        org_chart=evaluator_org_chart,
    )

    print(f"\n{'=' * 70}")
    print("Ablation Benchmark: generating fixed question set")
    print(f"{'=' * 70}")
    questions = evaluator.generate_benchmark_questions(
        n_questions=args.n_questions,
        question_types=args.question_types,
    )
    print(f"Generated {len(questions)} shared benchmark questions")

    llm_config = LLMConfig(model=args.model)
    runs: Dict[str, Dict[str, Any]] = {}
    question_payload = [question.to_dict() for question in questions]
    for config_name in args.configs:
        config = EXPERIMENTS[config_name]
        print(f"\n{'=' * 70}")
        print(f"Running config: {config.name}")
        print(config.description)
        print(f"{'=' * 70}")
        qa_system = build_qa_system(
            kg=kg,
            llm_config=llm_config,
            config=config,
            org_chart_path=args.org_chart,
        )
        benchmark = evaluator.evaluate_benchmark_questions(questions, qa_system=qa_system)
        runs[config.name] = {
            "config": asdict(config),
            "aggregate_metrics": benchmark.compute_aggregates(),
            "individual_results": [result.to_dict() for result in benchmark.individual_results],
        }
        write_output(args.output, args, question_payload, runs)
        print(f"Checkpoint saved after {config.name} -> {args.output}")

    write_output(args.output, args, question_payload, runs)
    print(f"\nSaved ablation results to {args.output}")


if __name__ == "__main__":
    main()
