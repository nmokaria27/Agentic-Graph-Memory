"""Build a governed open-world KG from prepared HotpotQA context documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from evaluation.hotpotqa.utils import context_text, load_prepared_examples
from multi_agent_kg.agents.base import ModelTier
from multi_agent_kg.core import (
    DeliberativeOrchestrator,
    DomainBuilder,
    GovernedKnowledgeGraph,
    LLMConfig,
    save_governed_kg,
)


CHECKPOINT_KEY = "_processed_doc_ids"


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "document"


def _hotpot_documents(examples_path: Path, *, supporting_only: bool, max_docs: int | None) -> List[Dict[str, Any]]:
    examples = load_prepared_examples(examples_path)
    wanted_titles: Set[str] | None = None
    if supporting_only:
        wanted_titles = set()
        for example in examples:
            for fact in example.supporting_facts:
                if isinstance(fact, (list, tuple)) and fact:
                    wanted_titles.add(str(fact[0]))

    docs: Dict[str, Dict[str, Any]] = {}
    for example in examples:
        for context in example.contexts:
            title = context.get("title", "Untitled")
            if wanted_titles is not None and title not in wanted_titles:
                continue
            if title in docs:
                docs[title]["metadata"]["question_ids"].append(example.question_id)
                continue
            text = f"Title: {title}\n\n{context_text(context)}"
            docs[title] = {
                "id": f"hotpot_{_slug(title)}",
                "text": text,
                "source": title,
                "metadata": {
                    "dataset": "hotpotqa",
                    "title": title,
                    "question_ids": [example.question_id],
                    "supporting_only": supporting_only,
                },
            }
    documents = list(docs.values())
    return documents[:max_docs] if max_docs is not None else documents


def _save_checkpoint(gkg: GovernedKnowledgeGraph, processed_ids: Set[str], path: Path) -> None:
    data = gkg.to_dict()
    data["stats"] = gkg.get_stats()
    data[CHECKPOINT_KEY] = sorted(processed_ids)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> Tuple[GovernedKnowledgeGraph, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GovernedKnowledgeGraph.from_dict(data), set(data.get(CHECKPOINT_KEY, []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", default="evaluation/results/hotpotqa_pilot_20.json")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--supporting-only", action="store_true")
    parser.add_argument("--governance-mode", default="triage", choices=["strict", "triage", "permissive", "audit_only"])
    parser.add_argument(
        "--reuse-corpus-schema",
        action="store_true",
        help=(
            "Reuse the first inferred schema across all documents. Off by default for HotpotQA "
            "because the corpus mixes unrelated Wikipedia topics."
        ),
    )
    parser.add_argument("--skip-evidence-linking", action="store_true", default=True)
    parser.add_argument("--skip-verification", action="store_true", default=True)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--output", default="evaluation/results/hotpotqa_governed_kg_20_supporting.json")
    parser.add_argument("--org-output", default="evaluation/results/hotpotqa_governed_kg_20_supporting_org.json")
    args = parser.parse_args()

    documents = _hotpot_documents(
        Path(args.examples_json),
        supporting_only=args.supporting_only,
        max_docs=args.max_docs,
    )
    checkpoint = Path(f"{args.output}.checkpoint.json")
    if args.resume_from_checkpoint and checkpoint.exists():
        governed_kg, processed_ids = _load_checkpoint(checkpoint)
        print(f"Resumed {checkpoint}: {len(processed_ids)} docs already processed", flush=True)
    else:
        governed_kg = GovernedKnowledgeGraph(governance_mode=args.governance_mode)
        processed_ids: Set[str] = set()

    remaining = [doc for doc in documents if doc["id"] not in processed_ids]
    llm_config = LLMConfig(model=args.model, temperature=0.2, max_tokens=4096)
    model_tiers = {tier: args.model for tier in ModelTier}
    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        governed_kg=governed_kg,
        governance_mode=args.governance_mode,
        reuse_corpus_schema=args.reuse_corpus_schema,
        expand_org_chart_with_schema=True,
        skip_evidence_linking=args.skip_evidence_linking,
        skip_verification=args.skip_verification,
        quality_threshold=0.4,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=True,
        enable_cross_document=True,
        enable_deliberation=False,
        model_tiers=model_tiers,
        schema_override=None,
    )

    failures: List[Dict[str, str]] = []
    print(f"HotpotQA governed KG: {len(documents)} docs total, {len(remaining)} remaining", flush=True)
    for i, doc in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] {doc['id']}", flush=True)
        try:
            orchestrator.process_document(
                text=doc["text"],
                source_path=doc["source"],
                document_id=doc["id"],
                metadata=doc["metadata"],
            )
            processed_ids.add(doc["id"])
        except Exception as exc:
            import traceback

            failures.append({"document_id": doc["id"], "error": str(exc), "traceback": traceback.format_exc()})
            print(f"ERROR {doc['id']}: {exc}", flush=True)
        if args.checkpoint_every and len(processed_ids) % args.checkpoint_every == 0:
            _save_checkpoint(governed_kg, processed_ids, checkpoint)
            print(f"checkpoint -> {checkpoint}", flush=True)

    if orchestrator.enable_cross_document:
        orchestrator._resolve_cross_document_entities()

    builder = DomainBuilder(llm_config)
    if not governed_kg.org_chart.domains and governed_kg.kg.entities:
        governed_kg.bootstrap_domains(builder)

    _save_checkpoint(governed_kg, processed_ids, checkpoint)
    save_governed_kg(governed_kg, args.output)
    Path(args.org_output).write_text(json.dumps(governed_kg.org_chart.to_dict(), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "processed": len(processed_ids),
                "failed": len(failures),
                "stats": governed_kg.get_stats(),
                "output": args.output,
                "org_output": args.org_output,
                "failures": failures,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
