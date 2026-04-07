"""
Runner script for KGAFE evaluation.

Usage:
    # Evaluate a single question against an existing KG
    python -m evaluation.kgafe.run_kgafe --kg-path kg_export.json --question "What causes X?"

    # Run full auto-benchmark (generates questions from KG, evaluates QA system)
    python -m evaluation.kgafe.run_kgafe --kg-path kg_export.json --benchmark --n-questions 30

    # Evaluate with org chart cache (uses existing QA system)
    python -m evaluation.kgafe.run_kgafe --kg-path kg_export.json --benchmark --org-chart org_chart_cache.json

    # Skip judge panel for faster evaluation (triple verification only)
    python -m evaluation.kgafe.run_kgafe --kg-path kg_export.json --benchmark --no-judge
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.kg_operations import load_kg
from evaluation.kgafe.evaluator import KGAFEEvaluator


def load_qa_system(kg, org_chart_path=None):
    """Load the QA system (QAOrchestrator) for benchmark evaluation."""
    from multi_agent_kg.core.domain_experts import QAOrchestrator, DomainBuilder, OrgChart

    llm_config = LLMConfig()

    if org_chart_path and os.path.exists(org_chart_path):
        # Load cached org chart
        print(f"Loading org chart from {org_chart_path}...")
        with open(org_chart_path) as f:
            cache = json.load(f)
        org_chart = OrgChart.from_dict(cache, kg)
    else:
        # Build org chart from scratch
        print("Building org chart from KG (this may take a while)...")
        builder = DomainBuilder(llm_config)
        org_chart = builder.build(kg)

    return QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=llm_config)


def main():
    parser = argparse.ArgumentParser(description="KGAFE: KG-Grounded Atomic Fact Evaluation")
    parser.add_argument("--kg-path", required=True, help="Path to kg_export.json")
    parser.add_argument("--question", help="Single question to evaluate")
    parser.add_argument("--answer", help="Answer to evaluate (required with --question)")
    parser.add_argument("--benchmark", action="store_true", help="Run full auto-benchmark")
    parser.add_argument("--n-questions", type=int, default=30, help="Number of benchmark questions")
    parser.add_argument("--org-chart", help="Path to cached org_chart_cache.json")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge panel (faster)")
    parser.add_argument("--output", help="Output path for results JSON")
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=["single_hop", "multi_hop", "aggregation", "comparison", "negative"],
        help="Question types to include in benchmark",
    )
    args = parser.parse_args()

    # Load KG
    print(f"Loading KG from {args.kg_path}...")
    kg = load_kg(args.kg_path)
    print(f"  Entities: {len(kg.entities)}, Triples: {len(kg.triples)}")

    # Initialize evaluator
    evaluator = KGAFEEvaluator(
        kg=kg,
        enable_judge_panel=not args.no_judge,
    )

    if args.question and args.answer:
        # Single question evaluation
        print(f"\nEvaluating single QA pair...")
        result = evaluator.evaluate_answer(args.question, args.answer)
        print(f"\nResults:")
        print(json.dumps(result.to_dict(), indent=2))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.benchmark:
        # Full benchmark
        qa_system = load_qa_system(kg, args.org_chart)
        results = evaluator.run_benchmark(
            n_questions=args.n_questions,
            qa_system=qa_system,
            question_types=args.question_types,
        )

        output_path = args.output or "evaluation/results/kgafe_benchmark.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nFull results saved to {output_path}")

    else:
        parser.error("Provide either --question + --answer, or --benchmark")


if __name__ == "__main__":
    main()
