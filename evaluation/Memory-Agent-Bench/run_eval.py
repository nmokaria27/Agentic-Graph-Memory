"""
Standalone MemoryAgentBench evaluator for Agent-Graph-Memory.

Does NOT require cloning MemoryAgentBench — downloads datasets via
HuggingFace hub and runs the inject-then-query loop directly.

Usage:
    # Quick smoke test (5 examples, EventQA):
    python evaluation/Memory-Agent-Bench/run_eval.py --dataset eventqa --max-samples 5

    # Full run, hybrid retrieval:
    python evaluation/Memory-Agent-Bench/run_eval.py \
        --dataset eventqa \
        --retrieval hybrid \
        --model gemma4:31b \
        --output evaluation/results/eventqa_hybrid.json

    # Run all supported datasets:
    python evaluation/Memory-Agent-Bench/run_eval.py --dataset all --output evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_graph_memory_adapter import AgentGraphMemoryWrapper


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = [
    "eventqa",       # Accurate Retrieval
    "ruler_qa",      # Accurate Retrieval
    "detectiveqa",   # Long-Range Understanding
    "fact_mh",       # Conflict Resolution
    "fact_sh",       # Conflict Resolution
]

# HF repo uses category names as splits; sub_dataset field filters within split.
# Actual split names: Accurate_Retrieval, Long_Range_Understanding, Conflict_Resolution
HF_DATASET_MAP = {
    "eventqa":     ("ai-hyz/MemoryAgentBench", "Accurate_Retrieval",      "eventqa"),
    "ruler_qa":    ("ai-hyz/MemoryAgentBench", "Accurate_Retrieval",      "ruler_qa"),
    "detectiveqa": ("ai-hyz/MemoryAgentBench", "Long_Range_Understanding", "detectiveqa"),
    "fact_mh":     ("ai-hyz/MemoryAgentBench", "Conflict_Resolution",      "fact_mh"),
    "fact_sh":     ("ai-hyz/MemoryAgentBench", "Conflict_Resolution",      "fact_sh"),
}


def _parse_questions(value) -> List[str]:
    """Parse questions field: JSON string or list → List[str]."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)]


def _parse_answers(value) -> List[List[str]]:
    """
    Parse answers field → List[List[str]] (one inner list per question,
    containing all valid answer strings for that question).

    Dataset format: JSON string of List[List[str]] where each inner list
    holds multiple acceptable answers for one question.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value.replace("'", '"'))
        except Exception:
            return [[value]]

    if not isinstance(value, list):
        return [[str(value)]]

    result = []
    for item in value:
        if isinstance(item, list):
            result.append([str(v) for v in item if v])
        elif isinstance(item, str):
            try:
                inner = json.loads(item.replace("'", '"'))
                result.append([str(v) for v in inner] if isinstance(inner, list) else [item])
            except Exception:
                result.append([item])
        else:
            result.append([str(item)])
    return result


def load_dataset_examples(dataset_name: str, max_samples: int = -1) -> List[Dict]:
    """
    Load examples from HuggingFace ai-hyz/MemoryAgentBench.
    Returns list of dicts with: context (str), questions (List[str]), answers (List[str]).
    """
    hf_repo, hf_split, sub_name = HF_DATASET_MAP[dataset_name]

    try:
        from datasets import load_dataset
        ds = load_dataset(hf_repo, split=hf_split)
        # Filter by sub_dataset name if field exists; otherwise take all
        if "sub_dataset" in ds.column_names:
            ds = ds.filter(lambda x: sub_name in (x.get("sub_dataset") or "").lower())
        examples_raw = list(ds)
    except Exception as exc:
        print(f"HuggingFace load failed ({exc}). Check local fallback path.")
        local = Path(__file__).parent / "data" / f"{dataset_name}.json"
        if not local.exists():
            raise FileNotFoundError(
                f"No local data at {local}. "
                f"Download from https://huggingface.co/datasets/{hf_repo}"
            ) from exc
        examples_raw = json.loads(local.read_text())

    # Normalise to {context, questions, answers}
    examples = []
    for raw in examples_raw:
        context = raw.get("context", "")
        if not context or len(context) < 50:
            continue
        questions = _parse_questions(raw.get("questions", []))
        answers   = _parse_answers(raw.get("answers", []))
        if not questions:
            continue
        # Pad answers to match questions length
        while len(answers) < len(questions):
            answers.append([""])
        examples.append({"context": context, "questions": questions, "answers": answers})

    if max_samples > 0:
        examples = examples[:max_samples]
    return examples


def chunk_text(text: str, chunk_size: int = 4096) -> List[str]:
    """Split text into ~chunk_size character chunks on sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _substring_em(prediction: str, gold: str) -> float:
    """Benchmark uses substring_exact_match for AR and CR tasks."""
    return 1.0 if _normalize(gold) in _normalize(prediction) else 0.0


def exact_match(prediction: str, golds: List[str]) -> float:
    """Max EM across all valid answers."""
    pred_norm = _normalize(prediction)
    return max(
        (1.0 if pred_norm == _normalize(g) else 0.0) for g in golds
    ) if golds else 0.0


def substring_exact_match(prediction: str, golds: List[str]) -> float:
    """Max substring EM across all valid answers (used for AR/CR datasets)."""
    pred_norm = _normalize(prediction)
    return max(
        (1.0 if _normalize(g) in pred_norm else 0.0) for g in golds
    ) if golds else 0.0


def token_f1(prediction: str, golds: List[str]) -> float:
    """Max token F1 across all valid answers."""
    pred_tokens = set(_normalize(prediction).split())
    if not pred_tokens or not golds:
        return 0.0
    best = 0.0
    for g in golds:
        gold_tokens = set(_normalize(g).split())
        if not gold_tokens:
            continue
        common = pred_tokens & gold_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def aggregate_metrics(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    primary  = sum(r["metrics"].get("primary", r["metrics"].get("exact_match", 0)) for r in rows) / n
    sub_em   = sum(r["metrics"].get("substring_exact_match", 0) for r in rows) / n
    em       = sum(r["metrics"].get("exact_match", 0) for r in rows) / n
    f1       = sum(r["metrics"].get("token_f1", 0) for r in rows) / n
    avg_build = sum(r["memory_construction_time"] for r in rows) / n
    avg_query = sum(r["query_time"] for r in rows) / n
    return {
        "accuracy": round(primary, 4),       # primary metric per benchmark spec
        "substring_exact_match": round(sub_em, 4),
        "exact_match": round(em, 4),
        "token_f1": round(f1, 4),
        "avg_memory_build_s": round(avg_build, 2),
        "avg_query_s": round(avg_query, 2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_dataset_eval(
    dataset_name: str,
    agent: AgentGraphMemoryWrapper,
    max_samples: int = -1,
    chunk_size: int = 4096,
    output_path: Optional[Path] = None,
    save_kg_dir: Optional[Path] = None,
    load_kg_dir: Optional[Path] = None,
    max_context_chars: Optional[int] = None,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    examples = load_dataset_examples(dataset_name, max_samples)
    print(f"Loaded {len(examples)} examples")

    rows: List[Dict] = []
    query_index = 0

    for ctx_idx, example in enumerate(examples):
        context = example.get("context", "")
        if not context or len(context) < 100:
            continue

        questions: List[str] = example.get("questions", [])
        # answers is List[List[str]] — one inner list per question with all valid answers
        answers: List[List[str]] = example.get("answers", [])

        # Subset to max_questions_per_context if context is huge (optional cap)
        max_q = int(os.environ.get("MAX_QUESTIONS_PER_CONTEXT", len(questions)))
        questions = questions[:max_q]
        answers = answers[:max_q]

        if max_context_chars and len(context) > max_context_chars:
            context = context[:max_context_chars]
            print(f"  [truncated to {max_context_chars:,} chars]")

        n_chunks = len(chunk_text(context, chunk_size))
        print(f"\n[{ctx_idx+1}/{len(examples)}] ctx_len={len(context):,} | chunks={n_chunks} | Qs={len(questions)}")

        # --- KG cache: load pre-built KG or run pipeline ---
        ctx_kg_path = load_kg_dir / f"context_{ctx_idx}" / "governed_kg.json" if load_kg_dir else None
        ctx_save_path = save_kg_dir / f"context_{ctx_idx}" / "governed_kg.json" if save_kg_dir else None

        t_mem_start = time.time()
        if ctx_kg_path and ctx_kg_path.exists():
            # Skip pipeline — load saved KG directly
            print(f"  Loading KG from cache: {ctx_kg_path}")
            agent._reset_context(ctx_idx)
            if agent.load_agent(str(ctx_kg_path.parent)):
                print(f"  KG loaded ({agent._governed_kg.get_stats().get('total_entities',0)} entities)")
            else:
                print("  Cache load failed — running pipeline")
                chunks = chunk_text(context, chunk_size)
                for chunk in chunks:
                    agent.send_message(chunk, memorizing=True, query_id=0, context_id=ctx_idx)
        else:
            # Run pipeline, optionally save result
            chunks = chunk_text(context, chunk_size)
            for chunk in chunks:
                agent.send_message(chunk, memorizing=True, query_id=0, context_id=ctx_idx)
            if ctx_save_path:
                agent._ensure_kg_built(ctx_idx)
                ctx_save_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_agent(str(ctx_save_path.parent))
                print(f"  KG saved to: {ctx_save_path.parent}")
        mem_time = time.time() - t_mem_start

        # AR/CR use substring_exact_match; LRU uses exact_match (per benchmark spec)
        _lru_datasets = {"detectiveqa"}
        use_substring = dataset_name not in _lru_datasets

        # --- Query phase ---
        for q_idx, (question, gold_list) in enumerate(zip(questions, answers)):
            result = agent.send_message(
                question, memorizing=False, query_id=query_index, context_id=ctx_idx
            )
            prediction = result.get("output", "")
            build_time = result.get("memory_construction_time", mem_time)
            query_time = result.get("query_time_len", 0.0)

            sub_em = substring_exact_match(prediction, gold_list)
            em     = exact_match(prediction, gold_list)
            f1     = token_f1(prediction, gold_list)
            # Primary metric matches benchmark spec
            primary = sub_em if use_substring else em

            row = {
                "context_id": ctx_idx,
                "query_id": query_index,
                "question": question,
                "gold_answers": gold_list,
                "prediction": prediction,
                "metrics": {
                    "substring_exact_match": sub_em,
                    "exact_match": em,
                    "token_f1": f1,
                    "primary": primary,
                },
                "memory_construction_time": build_time,
                "query_time": query_time,
            }
            rows.append(row)

            print(
                f"  Q{q_idx}: sub_em={sub_em:.0f} f1={f1:.3f} | "
                f"pred={prediction[:80]!r} | gold={gold_list[0]!r}"
            )

            if output_path:
                _save_incremental(output_path, dataset_name, rows)

            query_index += 1

    agg = aggregate_metrics(rows)
    print(f"\n--- {dataset_name} aggregate ---")
    print(json.dumps(agg, indent=2))

    return {"dataset": dataset_name, "aggregate": agg, "results": rows}


def _save_incremental(path: Path, dataset: str, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing[dataset] = {
        "aggregate": aggregate_metrics(rows),
        "results": rows,
    }
    path.write_text(json.dumps(existing, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MemoryAgentBench evaluation against Agent-Graph-Memory"
    )
    parser.add_argument(
        "--dataset",
        default="eventqa",
        choices=SUPPORTED_DATASETS + ["all"],
        help="Dataset to evaluate on (default: eventqa)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max examples per dataset (default: 5 for smoke test; -1 for full)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Characters per text chunk fed to the pipeline (default: 4096)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_DEFAULT_MODEL", os.environ.get("LLM_MODEL", "gemma4:31b")),
        help="LLM model name (default: gemma4:31b)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large"),
        help="Embedding model (default: mxbai-embed-large)",
    )
    parser.add_argument(
        "--retrieval",
        choices=["lexical", "dense", "hybrid", "graph_completion", "graph_summary", "chunk"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--answer-format",
        default="mab_substring",
        help="Answer-format profile (default: mab_substring). "
             "See evaluation/answer_format_profiles.py",
    )
    parser.add_argument(
        "--no-advanced-qa",
        action="store_true",
        help="Use basic QAOrchestrator instead of AdvancedQAOrchestrator",
    )
    parser.add_argument(
        "--output",
        default="evals/results/memoryagentbench_results.json",
        help="Output JSON path (default: evals/results/memoryagentbench_results.json)",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Truncate each context to this many characters before pipeline. "
             "Use 20000 for a ~5-chunk smoke test (~3-5 min). Default: no limit.",
    )
    parser.add_argument(
        "--save-kg-dir",
        default=None,
        help="Save built KGs to this directory so pipeline can be skipped on re-runs. "
             "Saved as {dir}/{dataset}/context_{idx}/governed_kg.json",
    )
    parser.add_argument(
        "--load-kg-dir",
        default=None,
        help="Load pre-built KGs from this directory instead of running pipeline. "
             "Must match layout written by --save-kg-dir.",
    )
    args = parser.parse_args()

    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    output_path = Path(args.output)

    print(f"Model:     {args.model}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Embedding: {args.embedding_model}")
    print(f"Datasets:  {datasets}")
    print(f"Samples:   {args.max_samples} per dataset")
    if args.load_kg_dir:
        print(f"KG cache:  LOAD from {args.load_kg_dir} (pipeline skipped)")
    elif args.save_kg_dir:
        print(f"KG cache:  SAVE to {args.save_kg_dir}")

    all_results = {}
    for dataset in datasets:
        save_kg_dir = Path(args.save_kg_dir) / dataset if args.save_kg_dir else None
        load_kg_dir = Path(args.load_kg_dir) / dataset if args.load_kg_dir else None

        agent = AgentGraphMemoryWrapper(
            model=args.model,
            embedding_model=args.embedding_model,
            retrieval_mode=args.retrieval,
            use_advanced_qa=not args.no_advanced_qa,
            answer_format=args.answer_format,
            verbose=True,
        )
        result = run_dataset_eval(
            dataset_name=dataset,
            agent=agent,
            max_samples=args.max_samples,
            chunk_size=args.chunk_size,
            output_path=output_path,
            save_kg_dir=save_kg_dir,
            load_kg_dir=load_kg_dir,
            max_context_chars=args.max_context_chars,
        )
        all_results[dataset] = result

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    summary = {d: r["aggregate"] for d, r in all_results.items()}
    print(json.dumps(summary, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
