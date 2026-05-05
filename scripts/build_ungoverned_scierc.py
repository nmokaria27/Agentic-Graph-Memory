#!/usr/bin/env python3
"""Build an ungoverned SciERC KG with the same extraction pipeline + config
used by build_governed_scierc.py / run_governed_vs_ungoverned.py's ungoverned path.

Produces an ungoverned KG JSON that mirrors the governed 50-doc triage artifact's
config (same split, same fixed-schema, same corpus-schema reuse, same model).

Supports checkpoint/resume so long runs can survive Ollama crashes: after every N
successfully processed documents the current KG and processed-doc-ID list are
written to ``<output>.checkpoint.json``. Passing ``--resume-from-checkpoint``
loads that file and skips documents that were already processed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

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


CHECKPOINT_KEY = "_processed_doc_ids"


def _clear_doc_caches(documents: Sequence[Dict[str, Any]]) -> None:
    results_dir = Path("evaluation/results")
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        cache_path = results_dir / f"{doc_id}.json"
        if cache_path.exists():
            cache_path.unlink()


def _save_checkpoint(
    kg: KnowledgeGraph,
    processed_ids: Set[str],
    failed: List[Dict[str, Any]],
    path: str,
) -> None:
    """Write the current KG plus processed-doc IDs atomically to disk."""
    payload = kg.to_dict() if hasattr(kg, "to_dict") else {"entities": [], "triples": []}
    payload[CHECKPOINT_KEY] = sorted(processed_ids)
    payload["_failed_documents"] = failed
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _load_checkpoint(path: str) -> Tuple[KnowledgeGraph, Set[str], List[Dict[str, Any]]]:
    """Rehydrate a KG and the processed-doc-ID set from a checkpoint."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    processed = set(data.get(CHECKPOINT_KEY, []))
    failed = data.get("_failed_documents", [])
    kg = KnowledgeGraph.from_dict(data) if hasattr(KnowledgeGraph, "from_dict") else KnowledgeGraph()
    return kg, processed, failed


def _filter_remaining(
    documents: Sequence[Dict[str, Any]],
    processed_ids: Set[str],
) -> List[Dict[str, Any]]:
    return [doc for doc in documents if doc.get("id") not in processed_ids]


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
    parser.add_argument(
        "--strict-source-only-verification",
        action="store_true",
        help=(
            "Ablation mode: EvidenceLinker/VerificationAgent only keep triples "
            "with exact source evidence spans from the current document."
        ),
    )
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save a checkpoint every N successfully processed documents (default: 5).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="If <output>.checkpoint.json exists, resume from it and skip already-processed docs.",
    )
    parser.add_argument(
        "--no-relation-gleaning",
        action="store_true",
        help="Disable the secondary relation-gleaning LLM pass (precision-favoring).",
    )
    parser.add_argument(
        "--no-fixed-schema-pairwise",
        action="store_true",
        help=(
            "Disable fixed-schema pairwise relation scoring. This avoids one "
            "extra LLM classification pass over sentence-local entity pairs "
            "and is the scalable setting for large SciERC builds."
        ),
    )
    args = parser.parse_args()

    scierc_path = os.path.join(args.data_dir, f"{args.split}.json")
    adapter = SciERCAdapter(scierc_path, skip_generic=True)
    documents = adapter.to_pipeline_input(max_docs=args.max_docs)

    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    model_tiers = {tier: args.model for tier in ModelTier}

    checkpoint_path = f"{args.output}.checkpoint.json"
    processed_ids: Set[str] = set()
    failed_documents: List[Dict[str, Any]] = []
    target: KnowledgeGraph

    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        target, processed_ids, failed_documents = _load_checkpoint(checkpoint_path)
        print(
            f"Resumed from checkpoint {checkpoint_path} — "
            f"{len(processed_ids)} docs already processed, "
            f"{len(target.entities)} entities, {len(target.triples)} triples carried over."
        )
    else:
        target = KnowledgeGraph()

    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        knowledge_graph=target,
        enable_governance=False,
        governance_mode="audit_only",
        reuse_corpus_schema=args.reuse_corpus_schema,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        strict_source_only_verification=args.strict_source_only_verification,
        quality_threshold=0.4,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=not args.fixed_schema,
        enable_fixed_schema_pairwise=not args.no_fixed_schema_pairwise,
        enable_cross_document=True,
        enable_deliberation=False,
        model_tiers=model_tiers,
        schema_override=SCIERC_SCHEMA if args.fixed_schema else None,
    )

    if args.no_relation_gleaning:
        orchestrator.relation_extractor.enable_relation_gleaning = False

    if args.clear_caches:
        _clear_doc_caches(documents)

    remaining = _filter_remaining(documents, processed_ids)

    print("=" * 72)
    print("  BUILD UNGOVERNED SCIERC KG")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Total documents in slice: {len(documents)}")
    print(f"Already processed (from checkpoint): {len(processed_ids)}")
    print(f"Documents to process this run: {len(remaining)}")
    print(f"Checkpoint every: {args.checkpoint_every} docs")
    print(f"Fixed schema: {args.fixed_schema}")
    print(f"Reuse corpus schema: {args.reuse_corpus_schema}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 72)

    started = time.perf_counter()
    newly_processed = 0

    print("=" * 70)
    print(f"PROCESSING CORPUS: {len(remaining)} documents (per-doc with checkpointing)")
    print("=" * 70)

    for i, doc in enumerate(remaining):
        print(f"\n[Document {i + 1}/{len(remaining)}]")
        try:
            orchestrator.process_document(
                text=doc.get("text"),
                source_path=doc.get("source"),
                document_id=doc.get("id"),
                metadata=doc.get("metadata"),
            )
            doc_id = doc.get("id")
            if doc_id is not None:
                processed_ids.add(doc_id)
            newly_processed += 1
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            failure = {
                "document_id": doc.get("id"),
                "error": str(exc),
                "traceback": tb,
            }
            failed_documents.append(failure)
            print(f"  ERROR: {failure['document_id']} failed: {failure['error']}")

        if args.checkpoint_every > 0 and newly_processed > 0 and newly_processed % args.checkpoint_every == 0:
            _save_checkpoint(target, processed_ids, failed_documents, checkpoint_path)
            print(
                f"  Checkpoint saved to {checkpoint_path} "
                f"({len(processed_ids)} docs, {len(target.triples)} triples)"
            )

    if orchestrator.enable_cross_document:
        print("\n" + "-" * 50)
        print("Cross-Document Entity Resolution")
        print("-" * 50)
        orchestrator._resolve_cross_document_entities()

    elapsed = time.perf_counter() - started

    # Save final checkpoint so a successful run leaves a consistent resume point.
    _save_checkpoint(target, processed_ids, failed_documents, checkpoint_path)

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
            "strict_source_only_verification": args.strict_source_only_verification,
            "governance_enabled": False,
            "relation_gleaning_enabled": getattr(
                orchestrator.relation_extractor, "enable_relation_gleaning", False
            ),
        },
        "entities": len(target.entities),
        "triples": len(target.triples),
        "processed_documents": len(processed_ids),
        "failed_documents": failed_documents,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_hours": round(elapsed / 3600, 3),
        "relation_funnel_summary": orchestrator.get_relation_funnel_summary(),
    }

    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  UNGOVERNED BUILD COMPLETE")
    print("=" * 72)
    print(f"Entities: {stats['entities']}")
    print(f"Triples: {stats['triples']}")
    print(f"Processed documents: {stats['processed_documents']}/{len(documents)}")
    print(f"Failed documents: {len(failed_documents)}")
    print(f"Elapsed: {stats['elapsed_seconds']}s  ({stats['elapsed_hours']}h)")
    print(f"Saved ungoverned KG to {args.output}")
    print(f"Checkpoint retained at {checkpoint_path}")
    if args.stats_output:
        print(f"Saved stats to {args.stats_output}")


if __name__ == "__main__":
    main()
