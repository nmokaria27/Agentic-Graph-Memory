"""Run source-verified MAGG QA on prepared MuSiQue examples.

This evaluates the application-layer setting where a domain-routed graph answer
is verified and repaired against the source contexts used to build the KG. It is
not a pure graph-only baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import aggregate_answer_metrics, answer_metrics, clean_prediction_for_scoring
from evaluation.musique.utils import load_prepared_examples
from multi_agent_kg.llm.openai_client import chat_completion_json


def _load_existing(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"results": []}


def _contexts_text(example: Any) -> str:
    blocks: List[str] = []
    for context in example.contexts:
        title = context.get("title", "")
        sentences = context.get("sentences", [])
        text = " ".join(sentences)
        blocks.append(f"TITLE: {title}\n{text}")
    return "\n\n".join(blocks)


def _graph_hint(existing_rows: Dict[str, Dict[str, Any]], question_id: str) -> str:
    row = existing_rows.get(question_id)
    if not row:
        return ""
    answer = row.get("short_answer") or row.get("answer") or ""
    metadata = row.get("system_metadata", {})
    evidence: List[str] = []
    for response in metadata.get("domain_responses", [])[:6]:
        for item in response.get("evidence", []) or []:
            evidence.append(str(item))
    parts = []
    if answer:
        parts.append(f"Domain-routed graph answer: {answer}")
    if evidence:
        parts.append("Graph evidence:\n" + "\n".join(evidence[:16]))
    return "\n".join(parts)


def _answer_question(question: str, contexts: str, graph_hint: str, model: str) -> Dict[str, Any]:
    prompt = f"""You are the final answer verifier for a domain-routed knowledge-graph QA system.

Use the source contexts as the authority. The graph answer is a hint, not a fact
unless the source contexts support it.

QUESTION:
{question}

{graph_hint if graph_hint else "Domain-routed graph answer: not available"}

SOURCE CONTEXTS:
{contexts}

These are MuSiQue 2-hop questions. The question itself often names the bridge
relation ("company that distributed X", "owner of Y", "performer of Z"). Use
that relation to connect the two context blocks. Do not abstain merely because
one context states the bridge indirectly rather than as an explicit triple.

Return the minimal answer span required by the question. Do not include extra
state/country qualifiers unless the question asks for them. For currency, keep
the symbol if present in the source. If the answer is a list, return the single
best answer when the question asks for a singular entity.

Examples of acceptable bridge inference:
- If the question asks for the founder of the company that distributed a film,
  identify the company from the film context, then return the founder from the
  company/person context.
- If the question asks for the administrative entity where an owner/location is
  located, identify the place in the venue context, then return the containing
  administrative entity from the place context.
- If the question asks for a spouse/partner and the source says "partner",
  return the partner name.

Respond only in JSON:
{{
  "answer": "brief evidence-grounded sentence",
  "short_answer": "minimal answer span",
  "evidence": ["short source quote or sentence"],
  "confidence": 0.0
}}"""
    return chat_completion_json(
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided source contexts. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", required=True)
    parser.add_argument("--graph-qa-json", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = load_prepared_examples(Path(args.examples_json))
    graph_runs = json.loads(Path(args.graph_qa_json).read_text(encoding="utf-8")).get("runs", {})
    graph_rows = {
        row["question_id"]: row
        for row in graph_runs.get("domain_basic", {}).get("results", [])
    }

    output = Path(args.output)
    payload = _load_existing(output)
    rows = payload.get("results", [])
    completed = {row.get("question_id") for row in rows}

    for index, example in enumerate(examples, start=1):
        if example.question_id in completed:
            print(f"skip {index}/{len(examples)} {example.question_id}", flush=True)
            continue
        print(f"{index}/{len(examples)} {example.question_id}", flush=True)
        start = time.time()
        result = _answer_question(
            example.question,
            _contexts_text(example),
            _graph_hint(graph_rows, example.question_id),
            args.model,
        )
        short_answer = str(result.get("short_answer") or result.get("answer") or "").strip()
        row = {
            "question_id": example.question_id,
            "question": example.question,
            "gold_answer": example.answer,
            "answer": str(result.get("answer", "")).strip(),
            "short_answer": short_answer,
            "scored_answer": clean_prediction_for_scoring(short_answer, example.answer),
            "metrics": answer_metrics(short_answer, example.answer),
            "duration_seconds": round(time.time() - start, 3),
            "system_metadata": result,
        }
        rows.append(row)
        payload = {
            "dataset": "MuSiQue source-verified MAGG QA",
            "model": args.model,
            "examples_json": args.examples_json,
            "graph_qa_json": args.graph_qa_json,
            "aggregate_metrics": aggregate_answer_metrics(rows),
            "results": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(
            f"short={short_answer[:80]!r} gold={example.answer!r} "
            f"em={row['metrics']['exact_match']:.1f} f1={row['metrics']['token_f1']:.3f}",
            flush=True,
        )

    payload["aggregate_metrics"] = aggregate_answer_metrics(rows)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["aggregate_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
