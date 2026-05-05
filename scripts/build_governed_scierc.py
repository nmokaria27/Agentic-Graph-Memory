#!/usr/bin/env python3
"""
Build a governed KG from a SciERC corpus slice.

This is the creation-side counterpart to the extraction evaluator:
instead of resetting for every document, it accumulates one governed KG,
bootstraps governance during creation, and exports the final structure.

Supports checkpoint/resume so long runs (e.g. the full 100-doc test set)
can survive Ollama crashes: after every N documents the current governed
KG and the list of processed doc IDs are written to
``<output>.checkpoint.json``. Passing ``--resume-from-checkpoint`` loads
that file and skips documents that were already processed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
from multi_agent_kg.agents.base import ModelTier


CHECKPOINT_KEY = "_processed_doc_ids"


def _clear_doc_caches(documents: List[Dict[str, Any]]) -> None:
    results_dir = Path("evaluation/results")
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        cache_path = results_dir / f"{doc_id}.json"
        if cache_path.exists():
            cache_path.unlink()


def _save_checkpoint(
    governed_kg: GovernedKnowledgeGraph,
    processed_ids: Set[str],
    path: str,
) -> None:
    """Write the current governed KG plus processed doc IDs to disk."""
    data = governed_kg.to_dict()
    data["stats"] = governed_kg.get_stats()
    data[CHECKPOINT_KEY] = sorted(processed_ids)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _load_checkpoint(path: str) -> Tuple[GovernedKnowledgeGraph, Set[str]]:
    """Rehydrate a governed KG and the processed-doc-ID set from a checkpoint."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    processed = set(data.get(CHECKPOINT_KEY, []))
    gkg = GovernedKnowledgeGraph.from_dict(data)
    return gkg, processed


def _filter_remaining(
    documents: List[Dict[str, Any]],
    processed_ids: Set[str],
) -> List[Dict[str, Any]]:
    return [doc for doc in documents if doc.get("id") not in processed_ids]


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
    parser.add_argument(
        "--clear-caches",
        action="store_true",
        help="Clear per-doc caches before running for fresh extraction.",
    )
    parser.add_argument(
        "--governance-mode",
        default="audit_only",
        choices=["strict", "triage", "permissive", "audit_only"],
    )
    parser.add_argument(
        "--output",
        default=os.path.join("evaluation", "results", "scierc_governed_created.json"),
    )
    parser.add_argument(
        "--org-output",
        default=os.path.join("evaluation", "results", "scierc_governed_created_org_chart.json"),
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional path to write run stats and extraction-funnel diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a checkpoint every N successfully processed documents (default: 10).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="If <output>.checkpoint.json exists, resume from it and skip already-processed docs.",
    )
    parser.add_argument(
        "--compare-bootstrap",
        action="store_true",
        help="After creation, rebuild a full org chart from the KG and compare assignment agreement.",
    )
    parser.add_argument(
        "--no-relation-gleaning",
        action="store_true",
        help="Disable the secondary relation-gleaning LLM pass (precision-favoring).",
    )
    parser.add_argument(
        "--fixed-schema-min-admission-confidence",
        type=float,
        default=0.75,
        help=(
            "Governance admission confidence floor for fixed-schema SciERC runs. "
            "Low-confidence triples are rejected with audit entries before admission."
        ),
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
    if args.clear_caches:
        _clear_doc_caches(documents)

    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    model_tiers = {tier: args.model for tier in ModelTier}

    checkpoint_path = f"{args.output}.checkpoint.json"
    processed_ids: Set[str] = set()
    governed_kg: GovernedKnowledgeGraph

    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        governed_kg, processed_ids = _load_checkpoint(checkpoint_path)
        if args.fixed_schema:
            governed_kg._min_admission_confidence = args.fixed_schema_min_admission_confidence
            governed_kg._confidence_policy_label = "fixed_schema_confidence_floor"
        print(
            f"Resumed from checkpoint {checkpoint_path} — "
            f"{len(processed_ids)} docs already processed, "
            f"{governed_kg.get_stats().get('entities', 0)} entities, "
            f"{governed_kg.get_stats().get('triples', 0)} triples carried over."
        )
    else:
        governed_kg = GovernedKnowledgeGraph(
            governance_mode=args.governance_mode,
            min_admission_confidence=(
                args.fixed_schema_min_admission_confidence if args.fixed_schema else None
            ),
            confidence_policy_label="fixed_schema_confidence_floor",
        )

    remaining = _filter_remaining(documents, processed_ids)

    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        governed_kg=governed_kg,
        governance_mode=args.governance_mode,
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

    print("=" * 72)
    print("  BUILD GOVERNED SCIERC KG")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Total documents in slice: {len(documents)}")
    print(f"Already processed (from checkpoint): {len(processed_ids)}")
    print(f"Documents to process this run: {len(remaining)}")
    print(f"Checkpoint every: {args.checkpoint_every} docs")
    print(f"Governance mode: {args.governance_mode}")
    print(f"Fixed schema: {args.fixed_schema}")
    print(f"Reuse corpus schema: {args.reuse_corpus_schema}")
    print()

    failed_documents: List[Dict[str, Any]] = []
    newly_processed = 0
    started = time.perf_counter()

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
            print(tb)

        if args.checkpoint_every > 0 and newly_processed > 0 and newly_processed % args.checkpoint_every == 0:
            _save_checkpoint(governed_kg, processed_ids, checkpoint_path)
            print(
                f"  Checkpoint saved to {checkpoint_path} "
                f"({len(processed_ids)} docs, "
                f"{governed_kg.get_stats().get('triples', 0)} triples)"
            )

    if orchestrator.enable_cross_document:
        print("\n" + "-" * 50)
        print("Cross-Document Entity Resolution")
        print("-" * 50)
        orchestrator._resolve_cross_document_entities()

    builder = DomainBuilder(llm_config)
    if not governed_kg.org_chart.domains:
        print("Bootstrapping final org chart from built KG...")
        governed_kg.bootstrap_domains(builder)

    if args.compare_bootstrap and governed_kg.kg.entities:
        print("Comparing bootstrap assignments against a full post-hoc domain build...")
        reference_org = governed_kg.org_chart
        final_org = builder.build(
            governed_kg.kg,
            target_num_domains=len(reference_org.domains) or None,
        )
        agreement = builder.compare_assignment_agreement(reference_org, final_org)
        governed_kg.set_bootstrap_assignment_stats(
            {
                **governed_kg.get_stats().get("bootstrap_assignment_stats", {}),
                "posthoc_assignment_agreement": agreement,
            }
        )

    # Save final checkpoint so a successful run leaves a consistent resume point.
    _save_checkpoint(governed_kg, processed_ids, checkpoint_path)

    save_governed_kg(governed_kg, args.output)
    # Record actual runtime config (model, tier mapping, ollama host) in the KG
    # JSON so future inspections aren't misled by --model that was ignored.
    try:
        with open(args.output, "r", encoding="utf-8") as handle:
            kg_payload = json.load(handle)
        kg_payload["runtime_config"] = {
            "model": args.model,
            "model_tiers": {t.value: m for t, m in model_tiers.items()},
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "governance_mode": args.governance_mode,
            "fixed_schema": args.fixed_schema,
            "reuse_corpus_schema": args.reuse_corpus_schema,
            "skip_evidence_linking": args.skip_evidence_linking,
            "skip_verification": args.skip_verification,
            "strict_source_only_verification": args.strict_source_only_verification,
            "clear_caches": args.clear_caches,
            "split": args.split,
            "max_docs": args.max_docs,
            "relation_gleaning_enabled": getattr(
                orchestrator.relation_extractor, "enable_relation_gleaning", False
            ),
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(kg_payload, handle, indent=2, default=str)
    except Exception as exc:
        print(f"WARN: could not stamp runtime_config into {args.output}: {exc}")
    with open(args.org_output, "w", encoding="utf-8") as handle:
        json.dump(governed_kg.org_chart.to_dict(), handle, indent=2)

    stats = governed_kg.get_stats()
    elapsed = time.perf_counter() - started
    run_stats = {
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
            "clear_caches": args.clear_caches,
            "governance_enabled": True,
            "governance_mode": args.governance_mode,
            "relation_gleaning_enabled": getattr(
                orchestrator.relation_extractor, "enable_relation_gleaning", False
            ),
        },
        "entities": stats.get("entities", 0),
        "triples": stats.get("triples", 0),
        "processed_documents": len(processed_ids),
        "failed_documents": failed_documents,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_hours": round(elapsed / 3600, 3),
        "relation_funnel_summary": orchestrator.get_relation_funnel_summary(),
        "kg_stats": stats,
    }
    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as handle:
            json.dump(run_stats, handle, indent=2, default=str)
    print("\nFinal governed KG stats:")
    print(json.dumps(stats, indent=2))
    print(f"Processed this run: {newly_processed}")
    print(f"Failed this run: {len(failed_documents)}")
    if failed_documents:
        print(json.dumps(failed_documents, indent=2))
    print(f"Saved governed KG to {args.output}")
    print(f"Saved org chart to {args.org_output}")
    if args.stats_output:
        print(f"Saved stats to {args.stats_output}")
    print(f"Checkpoint retained at {checkpoint_path}")


if __name__ == "__main__":
    main()
