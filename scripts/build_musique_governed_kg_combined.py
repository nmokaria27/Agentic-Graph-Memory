"""Build a governed MuSiQue KG by batching N supporting paragraphs into each LLM-processed document.

Trades per-paragraph extraction granularity for ~3-4× wall-clock speedup by amortizing per-doc
pipeline overhead (DomainClassifier, governance bootstrap, integration). Suitable for QA-oriented
KG builds where paragraph-level attribution is not required (the bridge expansion at QA time uses
the global KG).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from evaluation.musique.utils import load_prepared_examples
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
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "document"


def _supporting_paragraphs(examples_path: Path) -> List[Dict[str, Any]]:
    """Return deduped supporting paragraphs across all examples."""
    examples = load_prepared_examples(examples_path)
    seen: Dict[str, Dict[str, Any]] = {}
    for example in examples:
        sup_titles = set()
        for fact in example.supporting_facts:
            if isinstance(fact, (list, tuple)) and fact:
                sup_titles.add(str(fact[0]))
            elif isinstance(fact, dict):
                t = fact.get("title")
                if t:
                    sup_titles.add(str(t))
        for context in example.contexts:
            title = str(context.get("title", "Untitled"))
            if title not in sup_titles:
                continue
            if title in seen:
                seen[title]["question_ids"].append(example.question_id)
                continue
            text = " ".join(str(s) for s in context.get("sentences", []))
            seen[title] = {
                "title": title,
                "text": text,
                "question_ids": [example.question_id],
            }
    return list(seen.values())


def _batch(paragraphs: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    """Group paragraphs into combined documents."""
    docs: List[Dict[str, Any]] = []
    for i in range(0, len(paragraphs), batch_size):
        group = paragraphs[i:i + batch_size]
        body_parts = []
        titles = []
        question_ids: Set[str] = set()
        for p in group:
            titles.append(p["title"])
            question_ids.update(p["question_ids"])
            body_parts.append(f"=== Source: {p['title']} ===\n{p['text']}")
        body = "\n\n".join(body_parts)
        batch_idx = i // batch_size
        doc_id = f"musique_combined_b{batch_idx:03d}"
        docs.append({
            "id": doc_id,
            "text": body,
            "source": doc_id,
            "metadata": {
                "dataset": "musique",
                "combined": True,
                "batch_index": batch_idx,
                "batch_size": len(group),
                "titles": titles,
                "question_ids": sorted(question_ids),
                "supporting_only": True,
            },
        })
    return docs


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
    parser.add_argument("--examples-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--org-output", required=True)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of supporting paragraphs to combine per LLM-processed document.")
    parser.add_argument("--governance-mode", default="permissive",
                        choices=["strict", "triage", "permissive", "audit_only"])
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    paragraphs = _supporting_paragraphs(Path(args.examples_json))
    documents = _batch(paragraphs, batch_size=args.batch_size)
    print(f"prepared {len(paragraphs)} supporting paragraphs → {len(documents)} combined docs "
          f"(batch_size={args.batch_size})", flush=True)

    checkpoint = Path(f"{args.output}.checkpoint.json")
    if args.resume_from_checkpoint and checkpoint.exists():
        governed_kg, processed_ids = _load_checkpoint(checkpoint)
        print(f"Resumed {checkpoint}: {len(processed_ids)} combined docs already processed", flush=True)
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
        reuse_corpus_schema=False,
        expand_org_chart_with_schema=True,
        skip_evidence_linking=True,
        skip_verification=True,
        quality_threshold=0.35,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=True,
        enable_deterministic_value_harvesting=False,
        enable_deterministic_attribute_binding=False,
        enable_cross_document=True,
        enable_deliberation=False,
        model_tiers=model_tiers,
        target_num_domains=None,
        schema_override=None,
    )

    failures: List[Dict[str, str]] = []
    for index, doc in enumerate(remaining, start=1):
        print(f"[{index}/{len(remaining)}] {doc['id']} (titles={len(doc['metadata']['titles'])})",
              flush=True)
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
            failures.append({"document_id": doc["id"], "error": str(exc),
                             "traceback": traceback.format_exc()})
            print(f"ERROR {doc['id']}: {exc}", flush=True)
        if args.checkpoint_every and len(processed_ids) % args.checkpoint_every == 0:
            _save_checkpoint(governed_kg, processed_ids, checkpoint)
            print(f"checkpoint -> {checkpoint}", flush=True)

    if orchestrator.enable_cross_document:
        orchestrator._resolve_cross_document_entities()

    builder = DomainBuilder(llm_config, target_num_domains=None)
    if not governed_kg.org_chart.domains and governed_kg.kg.entities:
        governed_kg.bootstrap_domains(builder)
    governed_kg.org_chart.refresh_memory_cards(governed_kg.kg)

    _save_checkpoint(governed_kg, processed_ids, checkpoint)
    save_governed_kg(governed_kg, args.output)
    Path(args.org_output).write_text(
        json.dumps(governed_kg.org_chart.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "processed": len(processed_ids),
        "failed": len(failures),
        "combined_docs": len(documents),
        "source_paragraphs": len(paragraphs),
        "stats": governed_kg.get_stats(),
        "output": args.output,
        "org_output": args.org_output,
        "failures": failures,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
