"""
Compare enrichment with governance enabled vs disabled.

This is a lightweight experiment harness for the reframed paper:
given a base KG and new documents, run incremental enrichment in two modes
and report how many updates survive governance as well as the resulting KG size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from evaluation.adapters.scierc_adapter import SciERCAdapter
from evaluation.run_evaluation import SCIERC_SCHEMA
from multi_agent_kg.core import (
    GovernedKnowledgeGraph,
    IncrementalEnricher,
    LLMConfig,
    load_governed_kg,
)


def _snapshot_stats(governed_kg: GovernedKnowledgeGraph) -> Dict[str, Any]:
    stats = governed_kg.get_stats()
    return {
        "entities": stats.get("entities", 0),
        "triples": stats.get("triples", 0),
        "cross_domain_relations": stats.get("cross_domain_relations", 0),
        "audit_log_entries": stats.get("audit_log_entries", 0),
        "assignment_counts": stats.get("assignment_counts", {}),
        "decision_counts": stats.get("decision_counts", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run governed vs unguided enrichment experiment")
    parser.add_argument("--base-kg", required=True, help="Path to governed KG JSON")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "evaluation", "datasets", "scierc"),
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--skip-evidence-linking", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--fixed-schema", action="store_true")
    parser.add_argument("--reuse-corpus-schema", action="store_true", default=True)
    parser.add_argument(
        "--governance-review-mode",
        default="triage",
        choices=["triage", "strict", "audit_only", "permissive"],
    )
    parser.add_argument(
        "--disable-base-context",
        action="store_true",
        help="Extract enrichment docs against an empty working KG instead of a seeded copy of the base KG.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = SciERCAdapter(os.path.join(args.data_dir, f"{args.split}.json"), skip_generic=True)
    documents: List[Dict[str, Any]] = adapter.to_pipeline_input(max_docs=args.max_docs)
    llm_config = LLMConfig(model=args.model)

    baseline_gkg = load_governed_kg(args.base_kg)
    governed_gkg = GovernedKnowledgeGraph.from_dict(baseline_gkg.to_dict())
    unguided_gkg = GovernedKnowledgeGraph.from_dict(baseline_gkg.to_dict())

    governed_enricher = IncrementalEnricher(
        governed_kg=governed_gkg,
        llm_config=llm_config,
        enable_governance=True,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        governance_review_mode=args.governance_review_mode,
        use_base_context=not args.disable_base_context,
        fixed_schema=args.fixed_schema,
        reuse_corpus_schema=args.reuse_corpus_schema,
        schema_override=SCIERC_SCHEMA if args.fixed_schema else None,
    )
    unguided_enricher = IncrementalEnricher(
        governed_kg=unguided_gkg,
        llm_config=llm_config,
        enable_governance=False,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        use_base_context=not args.disable_base_context,
        fixed_schema=args.fixed_schema,
        reuse_corpus_schema=args.reuse_corpus_schema,
        schema_override=SCIERC_SCHEMA if args.fixed_schema else None,
    )

    governed_before = _snapshot_stats(governed_gkg)
    unguided_before = _snapshot_stats(unguided_gkg)
    governed_report = governed_enricher.add_documents(documents)
    unguided_report = unguided_enricher.add_documents(documents)
    governed_after = _snapshot_stats(governed_gkg)
    unguided_after = _snapshot_stats(unguided_gkg)

    payload = {
        "base_kg": args.base_kg,
        "split": args.split,
        "documents": len(documents),
        "governed": {
            "before": governed_before,
            "after": governed_after,
            "report": governed_report,
        },
        "unguided": {
            "before": unguided_before,
            "after": unguided_after,
            "report": unguided_report,
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(json.dumps(
        {
            "governed_new_triples_approved": governed_report.get("governance", {}).get("new_triples_approved", 0),
            "governed_new_triples_rejected": governed_report.get("governance", {}).get("new_triples_rejected", 0),
            "unguided_triples_added": unguided_report.get("merge_stats", {}).get("triples_added", 0),
        },
        indent=2,
    ))
    print(f"Saved enrichment experiment to {args.output}")


if __name__ == "__main__":
    main()
