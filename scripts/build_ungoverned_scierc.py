#!/usr/bin/env python3
"""Build an ungoverned SciERC KG with the same extraction pipeline + config
used by build_governed_scierc.py / run_governed_vs_ungoverned.py's ungoverned path.

Produces an ungoverned KG JSON that mirrors the governed 50-doc triage artifact's
config (same split, same fixed-schema, same corpus-schema reuse, same model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    KnowledgeGraph,
    LLMConfig,
    save_kg,
)
from multi_agent_kg.agents.base import ModelTier


def _clear_doc_caches(documents: Sequence[Dict[str, Any]]) -> None:
    results_dir = Path("evaluation/results")
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        cache_path = results_dir / f"{doc_id}.json"
        if cache_path.exists():
            cache_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ungoverned KG from SciERC docs")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=os.path.join("evaluation", "datasets", "scierc"))
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--fixed-schema", action="store_true")
    parser.add_argument("--reuse-corpus-schema", action="store_true", default=True)
    parser.add_argument("--skip-evidence-linking", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--clear-caches", action="store_true",
                        help="Clear per-doc caches before running for fresh extraction")
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "results", "ungoverned_created_50docs_v3.json"),
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional path to write run stats (elapsed time, aggregate)",
    )
    args = parser.parse_args()

    scierc_path = os.path.join(args.data_dir, f"{args.split}.json")
    adapter = SciERCAdapter(scierc_path, skip_generic=True)
    documents = adapter.to_pipeline_input(max_docs=args.max_docs)

    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    model_tiers = {tier: args.model for tier in ModelTier}
    target = KnowledgeGraph()
    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        knowledge_graph=target,
        enable_governance=False,
        governance_mode="audit_only",
        reuse_corpus_schema=args.reuse_corpus_schema,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        quality_threshold=0.4,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=not args.fixed_schema,
        enable_cross_document=True,
        enable_deliberation=False,
        model_tiers=model_tiers,
        schema_override=SCIERC_SCHEMA if args.fixed_schema else None,
    )

    if args.clear_caches:
        _clear_doc_caches(documents)

    print("=" * 72)
    print("  BUILD UNGOVERNED SCIERC KG")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Documents to process: {len(documents)}")
    print(f"Fixed schema: {args.fixed_schema}")
    print(f"Reuse corpus schema: {args.reuse_corpus_schema}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 72)

    started = time.perf_counter()
    aggregate = orchestrator.process_corpus(documents)
    elapsed = time.perf_counter() - started

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kg(target, str(output_path))

    stats: Dict[str, Any] = {
        "config": {
            "split": args.split,
            "max_docs": len(documents),
            "model": args.model,
            "model_tiers": {t.value: m for t, m in model_tiers.items()},
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "fixed_schema": args.fixed_schema,
            "reuse_corpus_schema": args.reuse_corpus_schema,
            "skip_evidence_linking": args.skip_evidence_linking,
            "skip_verification": args.skip_verification,
            "governance_enabled": False,
        },
        "entities": len(target.entities),
        "triples": len(target.triples),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_hours": round(elapsed / 3600, 3),
        "aggregate": aggregate,
    }

    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  UNGOVERNED BUILD COMPLETE")
    print("=" * 72)
    print(f"Entities: {stats['entities']}")
    print(f"Triples: {stats['triples']}")
    print(f"Elapsed: {stats['elapsed_seconds']}s  ({stats['elapsed_hours']}h)")
    print(f"Saved ungoverned KG to {args.output}")
    if args.stats_output:
        print(f"Saved stats to {args.stats_output}")


if __name__ == "__main__":
    main()
