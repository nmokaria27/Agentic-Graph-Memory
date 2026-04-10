#!/usr/bin/env python3
"""
Build a governed KG from a SciERC corpus slice.

This is the creation-side counterpart to the extraction evaluator:
instead of resetting for every document, it accumulates one governed KG,
bootstraps governance during creation, and exports the final structure.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from dotenv import load_dotenv

load_dotenv()

from evaluation.adapters.scierc_adapter import SciERCAdapter
from evaluation.run_evaluation import SCIERC_SCHEMA
from multi_agent_kg.core import (
    DeliberativeOrchestrator,
    DomainBuilder,
    GovernedKnowledgeGraph,
    LLMConfig,
    save_governed_kg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a governed KG from SciERC documents")
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test"],
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("evaluation", "datasets", "scierc"),
    )
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--model", default="gemma3:27b")
    parser.add_argument("--fixed-schema", action="store_true")
    parser.add_argument("--reuse-corpus-schema", action="store_true", default=True)
    parser.add_argument(
        "--governance-mode",
        default="audit_only",
        choices=["strict", "permissive", "audit_only"],
    )
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "results", "scierc_governed_created.json"),
    )
    parser.add_argument(
        "--org-output",
        default=os.path.join("evaluation", "results", "scierc_governed_created_org_chart.json"),
    )
    args = parser.parse_args()

    scierc_path = os.path.join(args.data_dir, f"{args.split}.json")
    adapter = SciERCAdapter(scierc_path, skip_generic=True)
    documents = adapter.to_pipeline_input(max_docs=args.max_docs)

    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    governed_kg = GovernedKnowledgeGraph(governance_mode=args.governance_mode)
    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        governed_kg=governed_kg,
        governance_mode=args.governance_mode,
        reuse_corpus_schema=args.reuse_corpus_schema,
        quality_threshold=0.4,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=not args.fixed_schema,
        enable_cross_document=True,
        enable_deliberation=False,
        schema_override=SCIERC_SCHEMA if args.fixed_schema else None,
    )

    print("=" * 72)
    print("  BUILD GOVERNED SCIERC KG")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Documents: {len(documents)}")
    print(f"Governance mode: {args.governance_mode}")
    print(f"Fixed schema: {args.fixed_schema}")
    print(f"Reuse corpus schema: {args.reuse_corpus_schema}")
    print()

    orchestrator.process_corpus(documents)

    if not governed_kg.org_chart.domains:
        print("Bootstrapping final org chart from built KG...")
        builder = DomainBuilder(llm_config)
        governed_kg.bootstrap_domains(builder)

    save_governed_kg(governed_kg, args.output)
    with open(args.org_output, "w", encoding="utf-8") as handle:
        json.dump(governed_kg.org_chart.to_dict(), handle, indent=2)

    stats = governed_kg.get_stats()
    print("\nFinal governed KG stats:")
    print(json.dumps(stats, indent=2))
    print(f"Saved governed KG to {args.output}")
    print(f"Saved org chart to {args.org_output}")


if __name__ == "__main__":
    main()
