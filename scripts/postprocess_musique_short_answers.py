"""Apply HotpotQA-style short-answer extraction to MuSiQue KG QA outputs.

The MuSiQue KG QA runner writes graph-grounded prose answers, while the gold
answers are short spans. This script applies the same extractor used in the
HotpotQA pipeline so EM/token-F1 are comparable to RAG (which is prompted for
short answers directly). The original raw `answer` is preserved; a new
`short_answer` field is added and `metrics` / `scored_answer` are recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import (
    aggregate_answer_metrics,
    answer_metrics,
    clean_prediction_for_scoring,
)
from multi_agent_kg.llm.openai_client import chat_completion


def _extract_short_answer(question: str, answer: str, *, model: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return ""
    lowered = answer.lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    abstain_starts = (
        "i am sorry",
        "i'm sorry",
        "sorry,",
        "unfortunately",
        "i cannot",
        "i can't",
        "i don't have",
        "there is no information",
        "there isn't",
        "no information is",
        "no evidence is",
    )
    if any(lowered.startswith(p) for p in abstain_starts):
        return ""

    prompt = f"""Extract the SHORT ANSWER SPAN from a verbose model answer. The short answer is the minimal substring that directly answers the question — typically 1 to 5 words.

Rules:
- Return ONLY the short answer span. No prefix, no explanation, no quotes, no punctuation at the end.
- If the answer names a person, place, organization, work, or other entity, return only the entity name.
- If the question expects a date, return only the date.
- If the question expects a number, return only the number.
- If the model answer is an abstention or genuinely does not answer the question, return an empty string.
- The short answer should be drawn DIRECTLY from the model answer's wording — do not paraphrase.

Examples:

Question: In which county is Kimbrough Memorial Stadium located?
Model answer: Kimbrough Memorial Stadium is located in Randall County, Texas.
Short answer: Randall County

Question: Who founded the company that distributed the film UHF?
Model answer: The company that distributed UHF was Orion Pictures, founded by Mike Medavoy and others.
Short answer: Mike Medavoy

Question: Who is the spouse of Caroline LeRoy?
Model answer: The spouse of Caroline LeRoy is Daniel Fletcher Webster, but there is no information provided regarding his children.
Short answer: Daniel Fletcher Webster

Question: What administrative territorial entity is the owner of Ciudad Deportiva located?
Model answer: Ciudad Deportiva is located in Tamaulipas, Mexico, within the municipality of Nuevo Laredo.
Short answer: Tamaulipas

Question: Did the team win in 1990?
Model answer: Yes, the team won the championship in 1990.
Short answer: yes

Question: What is the capital of France?
Model answer: I am sorry, but there is no information available about that.
Short answer:

Question: {question}
Model answer: {answer}
Short answer:"""
    try:
        result = chat_completion(
            [
                {"role": "system", "content": "Extract the minimal short-answer span from a verbose answer. Return ONLY the span, no other text."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=32,
        )
    except Exception:
        return answer

    lines = (result or "").strip().splitlines()
    if not lines:
        return answer
    short = lines[0].strip().strip('"').strip("'")
    for prefix in ("Answer:", "Short answer:", "Final answer:"):
        if short.lower().startswith(prefix.lower()):
            short = short[len(prefix):].strip()
    return short.rstrip(".") or answer


def _process_run(rows: List[Dict[str, Any]], *, model: str, label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        raw = row.get("answer", "") or ""
        gold = row.get("gold_answer", "")
        short = _extract_short_answer(row.get("question", ""), raw, model=model)
        new_row = dict(row)
        new_row["raw_answer"] = raw
        new_row["short_answer"] = short
        new_row["scored_answer"] = clean_prediction_for_scoring(short, gold)
        new_row["metrics"] = answer_metrics(short, gold)
        out.append(new_row)
        print(
            f"[{label}] {i}/{len(rows)} short={short[:60]!r} gold={gold!r} "
            f"em={new_row['metrics']['exact_match']:.1f} f1={new_row['metrics']['token_f1']:.3f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to musique_kg_qa_*.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--configs", nargs="*", help="Restrict to these config names")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    if "runs" in data:
        runs = data.get("runs", {})
        target = args.configs or list(runs.keys())
        for name in target:
            if name not in runs:
                print(f"skip missing: {name}", flush=True)
                continue
            rows = runs[name].get("results", [])
            new_rows = _process_run(rows, model=args.model, label=name)
            runs[name]["results"] = new_rows
            runs[name]["aggregate_metrics"] = aggregate_answer_metrics(new_rows)
        data["runs"] = runs
        summary = {n: r["aggregate_metrics"] for n, r in runs.items()}
    else:
        rows = data.get("results", [])
        label = data.get("system", Path(args.input).stem)
        new_rows = _process_run(rows, model=args.model, label=label)
        data["results"] = new_rows
        data["aggregate_metrics"] = aggregate_answer_metrics(new_rows)
        summary = {label: data["aggregate_metrics"]}

    data["postprocessed"] = True
    Path(args.output).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
