#!/usr/bin/env python3
"""
Run governed vs ungoverned KG creation on the same SciERC slice.

This is the central comparison for the reframed paper:
same documents, same extraction pipeline, same model, with and without the
governance/data-structure layer enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

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
    GovernedKnowledgeGraph,
    KnowledgeGraph,
    LLMConfig,
    save_governed_kg,
    save_kg,
)
from multi_agent_kg.agents.base import ModelTier


def _entity_to_dict(entity: Any) -> Dict[str, Any]:
    return {
        "id": entity.id,
        "labels": list(entity.labels),
        "type": entity.type,
        "metadata": dict(entity.metadata),
    }


def _triple_to_dict(triple: Any) -> Dict[str, Any]:
    return {
        "subject": triple.subject,
        "relation": triple.relation,
        "object": triple.object,
        "confidence": triple.confidence,
        "source": triple.source,
        "metadata": dict(triple.metadata),
    }


def _surface_entity(entity: Dict[str, Any]) -> str:
    if entity.get("text"):
        return str(entity["text"]).strip().lower()
    labels = entity.get("labels") or []
    if labels:
        return str(labels[0]).strip().lower()
    return str(entity.get("id", "")).strip().lower()


def _entity_set(entities: Sequence[Dict[str, Any]]) -> Set[str]:
    return {_surface_entity(entity) for entity in entities if _surface_entity(entity)}


def _surface_triple(triple: Dict[str, Any], entity_lookup: Dict[str, Dict[str, Any]]) -> Tuple[str, str, str]:
    meta = triple.get("metadata", {}) or {}
    subj = meta.get("original_subject")
    obj = meta.get("original_object")
    if not subj:
        subj_ent = entity_lookup.get(triple.get("subject", ""))
        subj = _surface_entity(subj_ent) if subj_ent else str(triple.get("subject", "")).strip().lower()
    if not obj:
        obj_ent = entity_lookup.get(triple.get("object", ""))
        obj = _surface_entity(obj_ent) if obj_ent else str(triple.get("object", "")).strip().lower()
    rel = str(triple.get("relation", "")).strip()
    return (str(subj).strip().lower(), rel, str(obj).strip().lower())


def _triple_set(entities: Sequence[Dict[str, Any]], triples: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    lookup = {entity.get("id"): entity for entity in entities}
    return {_surface_triple(triple, lookup) for triple in triples if triple.get("relation")}


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return round(len(a & b) / union, 4) if union else 0.0


def _clear_doc_caches(documents: Sequence[Dict[str, Any]]) -> None:
    results_dir = Path("evaluation/results")
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        cache_path = results_dir / f"{doc_id}.json"
        if cache_path.exists():
            cache_path.unlink()


def _run_creation(
    *,
    documents: List[Dict[str, Any]],
    llm_config: LLMConfig,
    model_tiers: Dict[ModelTier, str],
    fixed_schema: bool,
    reuse_corpus_schema: bool,
    skip_evidence_linking: bool,
    skip_verification: bool,
    governed: bool,
    governance_mode: str,
) -> Tuple[Dict[str, Any], float, Any]:
    if governed:
        target = GovernedKnowledgeGraph(governance_mode=governance_mode)
        orchestrator = DeliberativeOrchestrator(
            llm_config=llm_config,
            governed_kg=target,
            governance_mode=governance_mode,
            reuse_corpus_schema=reuse_corpus_schema,
            skip_evidence_linking=skip_evidence_linking,
            skip_verification=skip_verification,
            quality_threshold=0.4,
            max_refinement_iterations=1,
            enable_self_consistency=False,
            enable_open_world=not fixed_schema,
            enable_cross_document=True,
            enable_deliberation=False,
            model_tiers=model_tiers,
            schema_override=SCIERC_SCHEMA if fixed_schema else None,
        )
    else:
        target = KnowledgeGraph()
        orchestrator = DeliberativeOrchestrator(
            llm_config=llm_config,
            knowledge_graph=target,
            enable_governance=False,
            governance_mode="audit_only",
            reuse_corpus_schema=reuse_corpus_schema,
            skip_evidence_linking=skip_evidence_linking,
            skip_verification=skip_verification,
            quality_threshold=0.4,
            max_refinement_iterations=1,
            enable_self_consistency=False,
            enable_open_world=not fixed_schema,
            enable_cross_document=True,
            enable_deliberation=False,
            model_tiers=model_tiers,
            schema_override=SCIERC_SCHEMA if fixed_schema else None,
        )

    started = time.perf_counter()
    aggregate = orchestrator.process_corpus(documents)
    aggregate["relation_gleaning_enabled"] = getattr(
        orchestrator.relation_extractor, "enable_relation_gleaning", False
    )
    elapsed = time.perf_counter() - started
    return aggregate, elapsed, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare governed vs ungoverned KG creation")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=os.path.join("evaluation", "datasets", "scierc"))
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--fixed-schema", action="store_true")
    parser.add_argument("--reuse-corpus-schema", action="store_true", default=True)
    parser.add_argument("--skip-evidence-linking", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--governance-mode", default="audit_only", choices=["strict", "triage", "permissive", "audit_only"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--governed-kg-output", default="")
    parser.add_argument("--ungoverned-kg-output", default="")
    args = parser.parse_args()

    scierc_path = os.path.join(args.data_dir, f"{args.split}.json")
    adapter = SciERCAdapter(scierc_path, skip_generic=True)
    documents = adapter.to_pipeline_input(max_docs=args.max_docs)
    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    model_tiers = {tier: args.model for tier in ModelTier}

    _clear_doc_caches(documents)
    governed_aggregate, governed_elapsed, governed_gkg = _run_creation(
        documents=documents,
        llm_config=llm_config,
        model_tiers=model_tiers,
        fixed_schema=args.fixed_schema,
        reuse_corpus_schema=args.reuse_corpus_schema,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        governed=True,
        governance_mode=args.governance_mode,
    )

    _clear_doc_caches(documents)
    ungoverned_aggregate, ungoverned_elapsed, ungoverned_kg = _run_creation(
        documents=documents,
        llm_config=llm_config,
        model_tiers=model_tiers,
        fixed_schema=args.fixed_schema,
        reuse_corpus_schema=args.reuse_corpus_schema,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        governed=False,
        governance_mode=args.governance_mode,
    )

    governed_stats = governed_gkg.get_stats()
    governed_entities = [_entity_to_dict(entity) for entity in governed_gkg.kg.entities.values()]
    governed_triples = [_triple_to_dict(triple) for triple in governed_gkg.kg.triples]
    ungoverned_entities = [_entity_to_dict(entity) for entity in ungoverned_kg.entities.values()]
    ungoverned_triples = [_triple_to_dict(triple) for triple in ungoverned_kg.triples]

    governed_entity_set = _entity_set(governed_entities)
    ungoverned_entity_set = _entity_set(ungoverned_entities)
    governed_triple_set = _triple_set(governed_entities, governed_triples)
    ungoverned_triple_set = _triple_set(ungoverned_entities, ungoverned_triples)

    comparison = {
        "entity_overlap": _jaccard(governed_entity_set, ungoverned_entity_set),
        "triple_overlap": _jaccard(governed_triple_set, ungoverned_triple_set),
        "governance_overhead_seconds": round(governed_elapsed - ungoverned_elapsed, 2),
        "governance_overhead_pct": round(((governed_elapsed - ungoverned_elapsed) / ungoverned_elapsed) * 100, 2)
        if ungoverned_elapsed
        else 0.0,
        "unique_to_governed": len(governed_triple_set - ungoverned_triple_set),
        "unique_to_ungoverned": len(ungoverned_triple_set - governed_triple_set),
        "domain_coverage": governed_stats.get("domain_coverage", {}).get("fraction_entities_assigned", 0.0),
        "audit_completeness": round(
            governed_stats.get("audit_log_entries", 0) / governed_stats.get("triples", 1), 4
        ) if governed_stats.get("triples", 0) else 0.0,
        "cross_domain_fraction": round(
            governed_stats.get("cross_domain_relations", 0) / governed_stats.get("triples", 1), 4
        ) if governed_stats.get("triples", 0) else 0.0,
    }

    payload = {
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
            "governance_mode": args.governance_mode,
            "relation_gleaning_enabled": governed_aggregate.get("relation_gleaning_enabled", False),
        },
        "governed": {
            "entities": len(governed_entities),
            "triples": len(governed_triples),
            "domains": governed_stats.get("domains", 0),
            "cross_domain_relations": governed_stats.get("cross_domain_relations", 0),
            "entity_coverage_by_domain": governed_stats.get("domain_coverage", {}).get("fraction_entities_assigned", 0.0),
            "audit_log_entries": governed_stats.get("audit_log_entries", 0),
            "assignment_distribution": governed_stats.get("assignment_counts", {}),
            "decision_distribution": governed_stats.get("decision_counts", {}),
            "processing_time_seconds": round(governed_elapsed, 2),
            "aggregate": governed_aggregate,
            "relation_funnel_summary": governed_aggregate.get("relation_funnel_summary", {}),
        },
        "ungoverned": {
            "entities": len(ungoverned_entities),
            "triples": len(ungoverned_triples),
            "processing_time_seconds": round(ungoverned_elapsed, 2),
            "aggregate": ungoverned_aggregate,
            "relation_funnel_summary": ungoverned_aggregate.get("relation_funnel_summary", {}),
        },
        "comparison": comparison,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if args.governed_kg_output:
        save_governed_kg(governed_gkg, args.governed_kg_output)
    if args.ungoverned_kg_output:
        save_kg(ungoverned_kg, args.ungoverned_kg_output)

    print(json.dumps(payload["comparison"], indent=2))
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
