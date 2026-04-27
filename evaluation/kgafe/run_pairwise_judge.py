"""GraphRAG-style pairwise LLM-judge evaluation.

Reads QA answers already saved from a 4-way ablation run, presents pairs to a
judge LLM with four GraphRAG-style criteria (Comprehensiveness, Directness,
Faithfulness/Support, Usefulness), with order-swapped passes for position-bias
mitigation. Reports win/loss/tie rates per pair x dimension.

No QA generation happens here — only judging the saved answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from multi_agent_kg.llm.openai_client import chat_completion_json


DIMENSIONS = [
    ("comprehensiveness", "Comprehensiveness — does the answer cover all relevant aspects of the question?"),
    ("directness", "Directness — does the answer answer the question directly, without filler or evasion?"),
    ("faithfulness", "Faithfulness/Support — is every claim in the answer supported by the knowledge graph evidence shown?"),
    ("usefulness", "Usefulness — does the answer help the reader understand or reason about the topic?"),
]


JUDGE_SYSTEM = (
    "You are an impartial expert judge comparing two answers to a question about a "
    "scientific knowledge graph. Decide which answer is better on each criterion. "
    "Use only the question, the knowledge graph evidence, and the two answers. "
    "Do not be biased by which answer is shown first. If the answers are equally good "
    "on a criterion, return 'tie'. Always return valid JSON."
)


def _format_triples(triples: List[Dict[str, str]]) -> str:
    if not triples:
        return "(no supporting triples in KG — the gold answer indicates this is a negative question)"
    lines = []
    for t in triples:
        s = t.get("subject", "?")
        r = t.get("relation", "?")
        o = t.get("object", "?")
        lines.append(f"  ({s}) -[{r}]-> ({o})")
    return "\n".join(lines)


def build_judge_prompt(question: Dict[str, Any], ans_a: str, ans_b: str) -> str:
    crit_block = "\n".join(f"- {d}: {desc}" for d, desc in DIMENSIONS)
    triples_block = _format_triples(question.get("supporting_triples") or [])
    gold = question.get("gold_answer", "")
    qtext = question["question"]
    qtype = question.get("question_type", "")
    return (
        f"QUESTION ({qtype}):\n{qtext}\n\n"
        f"KNOWLEDGE GRAPH EVIDENCE (supporting triples):\n{triples_block}\n\n"
        f"GOLD REFERENCE ANSWER:\n{gold}\n\n"
        f"ANSWER A:\n{ans_a}\n\n"
        f"ANSWER B:\n{ans_b}\n\n"
        f"Evaluate the two answers on these criteria:\n{crit_block}\n\n"
        "For each criterion, pick the better answer (\"A\", \"B\", or \"tie\") and give a "
        "one-sentence reason. Also pick an overall winner.\n\n"
        "Respond with this JSON shape and nothing else:\n"
        "{\n"
        "  \"comprehensiveness\": {\"winner\": \"A|B|tie\", \"reason\": \"...\"},\n"
        "  \"directness\": {\"winner\": \"A|B|tie\", \"reason\": \"...\"},\n"
        "  \"faithfulness\": {\"winner\": \"A|B|tie\", \"reason\": \"...\"},\n"
        "  \"usefulness\": {\"winner\": \"A|B|tie\", \"reason\": \"...\"},\n"
        "  \"overall\": {\"winner\": \"A|B|tie\", \"reason\": \"...\"}\n"
        "}"
    )


def judge_one(question: Dict[str, Any], ans_a: str, ans_b: str, model: str) -> Dict[str, Any]:
    user = build_judge_prompt(question, ans_a, ans_b)
    try:
        result = chat_completion_json(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=0.0,
        )
    except Exception as e:
        return {"_error": str(e)}
    return result if isinstance(result, dict) else {"_error": f"non-dict result: {type(result).__name__}"}


def normalize_winner(raw: Any) -> str:
    if not isinstance(raw, str):
        return "tie"
    s = raw.strip().lower()
    if s.startswith("a"):
        return "A"
    if s.startswith("b"):
        return "B"
    return "tie"


def aggregate(verdicts: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """verdicts is list of (orientation, raw_judge_dict) where orientation is 'AB' or 'BA'.
    In 'AB' orientation, the original config A was shown as 'A'; in 'BA', original A was shown as 'B'.
    Returns counts per dimension keyed to the ORIGINAL config A vs B."""
    out: Dict[str, Dict[str, int]] = {
        d: {"a_wins": 0, "b_wins": 0, "ties": 0} for d, _ in DIMENSIONS
    }
    out["overall"] = {"a_wins": 0, "b_wins": 0, "ties": 0}
    for orientation, v in verdicts:
        if not isinstance(v, dict) or "_error" in v:
            continue
        for dim in list(out.keys()):
            entry = v.get(dim) or {}
            w = normalize_winner(entry.get("winner"))
            if orientation == "AB":
                if w == "A":
                    out[dim]["a_wins"] += 1
                elif w == "B":
                    out[dim]["b_wins"] += 1
                else:
                    out[dim]["ties"] += 1
            else:  # BA: 'A' in judge view = original B
                if w == "A":
                    out[dim]["b_wins"] += 1
                elif w == "B":
                    out[dim]["a_wins"] += 1
                else:
                    out[dim]["ties"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="evaluation/results/scierc_50docs_qa_ablation_with_graphrag.json",
        help="Path to the saved 4-way ablation answers.",
    )
    ap.add_argument(
        "--output",
        default="evaluation/results/scierc_50docs_qa_pairwise_judged.json",
    )
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=[
            "domain_basic:flat_path_basic",
            "domain_basic:graphrag_basic",
            "domain_advanced:flat_path_basic",
            "domain_advanced:graphrag_basic",
            "graphrag_basic:flat_path_basic",
        ],
        help="config_a:config_b pairs to judge.",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    questions = {q["question_id"]: q for q in data["questions"]}
    runs = data["runs"]
    answers: Dict[str, Dict[str, str]] = {}
    for cfg, run in runs.items():
        per_q = {}
        for r in run["individual_results"]:
            per_q[r["question_id"]] = r["answer"]
        answers[cfg] = per_q

    pair_specs = [tuple(p.split(":", 1)) for p in args.pairs]
    for a, b in pair_specs:
        if a not in answers or b not in answers:
            print(f"FATAL: missing config in answers: {a} or {b}", file=sys.stderr)
            return 2

    qids = sorted(questions.keys())
    print(f"Judging {len(pair_specs)} pairs x {len(qids)} questions x 2 orderings = {len(pair_specs) * len(qids) * 2} judge calls", flush=True)

    pair_records: List[Dict[str, Any]] = []
    for cfg_a, cfg_b in pair_specs:
        print(f"\n=== {cfg_a} vs {cfg_b} ===", flush=True)
        per_q_verdicts: List[Dict[str, Any]] = []
        verdicts_for_agg: List[Tuple[str, Dict[str, Any]]] = []
        for qid in qids:
            q = questions[qid]
            ans_a = answers[cfg_a].get(qid, "")
            ans_b = answers[cfg_b].get(qid, "")
            print(f"  [{qid}] AB...", end="", flush=True)
            v_ab = judge_one(q, ans_a, ans_b, args.model)
            print(" BA...", end="", flush=True)
            v_ba = judge_one(q, ans_b, ans_a, args.model)
            print(" done", flush=True)
            per_q_verdicts.append({
                "question_id": qid,
                "question_type": q.get("question_type"),
                "verdict_AB": v_ab,
                "verdict_BA": v_ba,
            })
            verdicts_for_agg.append(("AB", v_ab))
            verdicts_for_agg.append(("BA", v_ba))

        agg = aggregate(verdicts_for_agg)
        # Win-rate over all comparisons (votes), excluding ties
        summary = {}
        for dim, counts in agg.items():
            total = counts["a_wins"] + counts["b_wins"] + counts["ties"]
            decisive = counts["a_wins"] + counts["b_wins"]
            summary[dim] = {
                **counts,
                "total": total,
                "a_win_rate": (counts["a_wins"] / total) if total else 0.0,
                "b_win_rate": (counts["b_wins"] / total) if total else 0.0,
                "tie_rate": (counts["ties"] / total) if total else 0.0,
                "a_win_rate_decisive": (counts["a_wins"] / decisive) if decisive else 0.0,
            }

        pair_records.append({
            "config_a": cfg_a,
            "config_b": cfg_b,
            "summary": summary,
            "per_question": per_q_verdicts,
        })

        # Pretty print the summary inline
        print(f"  --- summary {cfg_a} (A) vs {cfg_b} (B) ---")
        for dim in [d for d, _ in DIMENSIONS] + ["overall"]:
            s = summary[dim]
            print(
                f"    {dim:18s}  A wins {s['a_wins']:>3} / B wins {s['b_wins']:>3} / ties {s['ties']:>3}  "
                f"(A win-rate {s['a_win_rate']*100:.1f}%, decisive {s['a_win_rate_decisive']*100:.1f}%)"
            )

    out = {
        "model": args.model,
        "input_file": args.input,
        "n_questions": len(qids),
        "pairs": pair_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
