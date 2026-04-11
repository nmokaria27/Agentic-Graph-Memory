#!/usr/bin/env python3
"""
End-to-end evaluation runner for the multi-agent KG extraction pipeline
on the SciERC benchmark dataset.

Workflow:
    1. Load SciERC test set via the adapter
    2. Convert documents to pipeline input format
    3. (Optionally) run the pipeline on each document
    4. Evaluate pipeline output against gold standard
    5. Print summary metrics

Usage examples:

    # Full pipeline run on 5 documents:
    python evaluation/run_evaluation.py --max-docs 5

    # Evaluate pre-computed results only:
    python evaluation/run_evaluation.py --skip-pipeline --results-dir evaluation/results/

    # Use a specific SciERC split:
    python evaluation/run_evaluation.py --split dev --max-docs 10

    # Save metrics to JSON:
    python evaluation/run_evaluation.py --skip-pipeline --results-dir evaluation/results/ --output-json metrics.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on the path
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVAL_DIR)

from adapters.scierc_adapter import SciERCAdapter
from evaluate_kg import evaluate_corpus, print_report


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

# SciERC fixed schema for benchmark evaluation
SCIERC_SCHEMA = {
    "domain": "ScientificInformationExtraction",
    "description": "Scientific entity and relation extraction following SciERC schema",
    "entity_types": [
        {"type": "TASK", "description": "A scientific task, problem, or application (e.g., 'named entity recognition', 'machine translation', 'image classification')", "priority": "high"},
        {"type": "METHOD", "description": "A scientific method, algorithm, model, system, or tool (e.g., 'neural network', 'SVM', 'LSTM', 'Amorph')", "priority": "high"},
        {"type": "METRIC", "description": "An evaluation metric or measure (e.g., 'F1 score', 'accuracy', 'BLEU score')", "priority": "high"},
        {"type": "MATERIAL", "description": "A dataset, corpus, language, or other input resource (e.g., 'Penn Treebank', 'Japanese text', 'ImageNet')", "priority": "high"},
        {"type": "OTHERSCIENTIFICTERM", "description": "Other scientific terms, concepts, features, or phenomena that don't fit the above (e.g., 'word embeddings', 'parse tree', 'named entities', 'lexical features')", "priority": "high"},
    ],
    "relation_types": [
        {"type": "Used-for", "description": "X is used for Y (method applied to task, tool used for purpose)", "priority": "high"},
        {"type": "Feature-of", "description": "X is a feature/property/attribute of Y", "priority": "high"},
        {"type": "Part-of", "description": "X is a part/component/subset of Y", "priority": "high"},
        {"type": "Compare", "description": "X is compared to Y (methods compared, baselines)", "priority": "high"},
        {"type": "Hyponym-of", "description": "X is a type/subclass/instance of Y", "priority": "medium"},
        {"type": "Conjunction", "description": "X and Y are coordinated/co-listed (used together)", "priority": "medium"},
        {"type": "Evaluate-for", "description": "X is evaluated/measured for Y (metric applied to task)", "priority": "medium"},
    ],
}


def run_pipeline_on_docs(
    documents: List[Dict[str, Any]],
    model: str = "gemma4:31b",
    output_dir: str = "evaluation/results",
    schema_override: Optional[Dict[str, Any]] = None,
    reuse_corpus_schema: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the multi-agent KG pipeline on a list of documents.

    Args:
        documents: Pipeline-formatted documents (id, text, metadata).
        model: LLM model to use.
        output_dir: Directory to save per-document results.
        schema_override: If provided, use fixed schema instead of dynamic discovery.

    Returns:
        Mapping of doc_key -> {"entities": [...], "triples": [...]}.
    """
    from multi_agent_kg.core import LLMConfig, DeliberativeOrchestrator, KnowledgeGraph

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for i, doc in enumerate(documents):
        doc_key = doc["id"]
        output_file = os.path.join(output_dir, f"{doc_key}.json")

        # Skip if already computed
        if os.path.exists(output_file):
            print(f"  [{i+1}/{len(documents)}] {doc_key}: loading cached result")
            with open(output_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            kg_data = data.get("knowledge_graph", data)
            results[doc_key] = kg_data
            continue

        print(f"  [{i+1}/{len(documents)}] {doc_key}: running pipeline...")
        t0 = time.time()

        try:
            llm_config = LLMConfig(
                model=model,
                temperature=0.2,
                max_tokens=4096,
            )
            orchestrator = DeliberativeOrchestrator(
                llm_config=llm_config,
                knowledge_graph=KnowledgeGraph(),
                governance_mode="audit_only",
                reuse_corpus_schema=reuse_corpus_schema,
                quality_threshold=0.4,
                max_refinement_iterations=1,
                enable_self_consistency=False,
                enable_open_world=False if schema_override else True,
                enable_cross_document=False,
                schema_override=schema_override,
            )
            orchestrator.process_corpus([doc])
            export = orchestrator.export()
            kg_data = export["knowledge_graph"]

            # Save to file
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(export, fh, indent=2, default=str)

            results[doc_key] = kg_data
            elapsed = time.time() - t0
            n_ents = len(kg_data.get("entities", []))
            n_trips = len(kg_data.get("triples", []))
            print(f"    -> {n_ents} entities, {n_trips} triples ({elapsed:.1f}s)")

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    -> ERROR after {elapsed:.1f}s: {exc}")
            # Save empty result so we don't retry
            empty = {"knowledge_graph": {"entities": [], "triples": []}}
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(empty, fh, indent=2)
            results[doc_key] = {"entities": [], "triples": []}

    return results


def load_precomputed_results(
    results_dir: str, doc_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load pre-computed pipeline results from a directory.

    Args:
        results_dir: Path to directory containing per-document JSON files.
        doc_keys: If given, only load results for these document keys.

    Returns:
        Mapping of doc_key -> {"entities": [...], "triples": [...]}.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(results_dir):
        print(f"  WARNING: results directory not found: {results_dir}")
        return results

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        doc_key = fname.replace(".json", "")
        if doc_keys and doc_key not in doc_keys:
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        kg_data = data.get("knowledge_graph", data)
        results[doc_key] = kg_data

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation of the KG pipeline on SciERC."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="SciERC data split to use (default: test)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(EVAL_DIR, "datasets", "scierc"),
        help="Directory containing SciERC JSON files",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit to first N documents (useful for testing)",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running the pipeline; only evaluate pre-computed results",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(EVAL_DIR, "results"),
        help="Directory for per-document pipeline results",
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b",
        help="LLM model name to use (default: gemma4:31b)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy matching (default: 0.8)",
    )
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Include Generic-type entities in evaluation",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write metrics to a JSON file",
    )
    parser.add_argument(
        "--fixed-schema",
        action="store_true",
        help="Use SciERC's fixed entity/relation schema instead of dynamic discovery",
    )
    parser.add_argument(
        "--reuse-corpus-schema",
        action="store_true",
        help="Reuse the first discovered schema across documents for speed and consistency",
    )
    args = parser.parse_args()

    # --- Step 1: Load SciERC ---
    scierc_path = os.path.join(args.data_dir, f"{args.split}.json")
    print("=" * 72)
    print("  SciERC EVALUATION RUNNER")
    print("=" * 72)
    print()
    print(f"Loading SciERC {args.split} split from: {scierc_path}")
    adapter = SciERCAdapter(scierc_path, skip_generic=not args.include_generic)
    print(f"  {len(adapter)} documents loaded")

    # --- Step 2: Convert to pipeline format ---
    pipeline_docs = adapter.to_pipeline_input(max_docs=args.max_docs)
    gold_data = adapter.get_all_gold(max_docs=args.max_docs)
    doc_keys = [g["doc_key"] for g in gold_data]

    total_gold_ents = sum(len(g["entities"]) for g in gold_data)
    total_gold_trips = sum(len(g["triples"]) for g in gold_data)
    print(f"  Evaluating {len(gold_data)} documents")
    print(f"  Total gold entities: {total_gold_ents}")
    print(f"  Total gold triples:  {total_gold_trips}")
    print()

    # --- Step 3: Run pipeline or load results ---
    if args.skip_pipeline:
        print(f"Loading pre-computed results from: {args.results_dir}")
        pred_docs = load_precomputed_results(args.results_dir, doc_keys)
        print(f"  {len(pred_docs)} document results loaded")
    else:
        schema = SCIERC_SCHEMA if args.fixed_schema else None
        if schema:
            print("Using FIXED SciERC schema (entity types: Task, Method, Metric, Material, OtherScientificTerm)")
        print("Running pipeline on documents...")
        pred_docs = run_pipeline_on_docs(
            pipeline_docs,
            model=args.model,
            output_dir=args.results_dir,
            schema_override=schema,
            reuse_corpus_schema=args.reuse_corpus_schema,
        )
    print()

    # --- Step 4: Evaluate ---
    print("Computing evaluation metrics...")
    metrics = evaluate_corpus(gold_data, pred_docs, args.fuzzy_threshold)
    print()

    # --- Step 5: Print report ---
    print_report(metrics)

    # --- Optional: save JSON ---
    if args.output_json:
        from evaluate_kg import PRF
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, PRF):
                serializable[k] = {
                    "precision": v.precision(),
                    "recall": v.recall(),
                    "f1": v.f1(),
                    "tp": v.tp, "fp": v.fp, "fn": v.fn,
                }
            elif isinstance(v, dict):
                serializable[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, PRF):
                        serializable[k][kk] = {
                            "precision": vv.precision(),
                            "recall": vv.recall(),
                            "f1": vv.f1(),
                            "tp": vv.tp, "fp": vv.fp, "fn": vv.fn,
                        }
                    else:
                        serializable[k][kk] = vv
            else:
                serializable[k] = v
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
        print(f"\nMetrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
