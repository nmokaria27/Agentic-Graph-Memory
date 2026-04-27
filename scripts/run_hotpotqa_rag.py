"""Run a document-RAG baseline on a prepared HotpotQA pilot set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import (
    aggregate_answer_metrics,
    answer_metrics,
    clean_prediction_for_scoring,
    context_text,
    lexical_retrieve,
    load_prepared_examples,
)
from multi_agent_kg.llm.openai_client import chat_completion


def _build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, context in enumerate(contexts, start=1):
        blocks.append(f"[{i}] {context.get('title', 'Untitled')}\n{context_text(context)}")
    return (
        "Answer the HotpotQA question using only the retrieved context below.\n"
        "Return only the short answer string, not JSON and not a full explanation. "
        "If the answer is yes or no, return only yes or no.\n\n"
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        + "\n\n".join(blocks)
        + "\n\nShort answer:"
    )


def _clean_short_answer(text: str) -> str:
    answer = (text or "").strip()
    if not answer:
        return ""
    first_line = answer.splitlines()[0].strip()
    for prefix in ("Answer:", "Short answer:", "Final answer:"):
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    return first_line.strip().strip('"').strip("'")


def _answer_question(example, *, model: str, top_k: int) -> Dict[str, Any]:
    retrieved = lexical_retrieve(example.question, example.contexts, k=top_k)
    start = time.time()
    result = chat_completion(
        [
            {
                "role": "system",
                "content": "You answer HotpotQA questions from supplied evidence with concise gold-style answers. Return only the answer string.",
            },
            {"role": "user", "content": _build_prompt(example.question, retrieved)},
        ],
        model=model,
        temperature=0.0,
        # Gemma4 can spend many tokens in hidden reasoning before emitting
        # visible content through Ollama's OpenAI-compatible endpoint.
        max_tokens=512,
    )
    duration = time.time() - start
    answer = _clean_short_answer(result)
    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_answer": example.answer,
        "answer": answer.strip(),
        "scored_answer": clean_prediction_for_scoring(answer, example.answer),
        "raw_answer": result,
        "retrieved_titles": [context.get("title", "") for context in retrieved],
        "metrics": answer_metrics(answer, example.answer),
        "duration_seconds": round(duration, 3),
    }


def _load_existing(output: Path) -> List[Dict[str, Any]]:
    if not output.exists():
        return []
    data = json.loads(output.read_text(encoding="utf-8"))
    return data.get("results", [])


def _write_output(output: Path, *, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "HotpotQA distractor dev pilot",
        "system": "rag_basic",
        "model": args.model,
        "top_k": args.top_k,
        "examples_json": args.examples_json,
        "aggregate_metrics": aggregate_answer_metrics(rows),
        "results": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", default="evaluation/results/hotpotqa_pilot_20.json")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", default="evaluation/results/hotpotqa_rag_20.json")
    args = parser.parse_args()

    examples = load_prepared_examples(Path(args.examples_json))
    output = Path(args.output)
    rows = _load_existing(output)
    completed = {row.get("question_id") for row in rows}

    for i, example in enumerate(examples, start=1):
        if example.question_id in completed:
            print(f"[{i}/{len(examples)}] skip {example.question_id}", flush=True)
            continue
        print(f"[{i}/{len(examples)}] rag_basic {example.question_id}: {example.question}", flush=True)
        row = _answer_question(example, model=args.model, top_k=args.top_k)
        rows.append(row)
        _write_output(output, args=args, rows=rows)
        print(
            f"[{i}/{len(examples)}] answer={row['answer']!r} gold={row['gold_answer']!r} "
            f"em={row['metrics']['exact_match']:.1f} f1={row['metrics']['token_f1']:.3f}",
            flush=True,
        )

    _write_output(output, args=args, rows=rows)
    print(json.dumps(aggregate_answer_metrics(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
