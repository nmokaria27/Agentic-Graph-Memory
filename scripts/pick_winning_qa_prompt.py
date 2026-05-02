"""Pick the winning prompt config per QA cell from dev results.

Inputs: multiple JSON files from `run_musique_kg_qa.py` produced under different
prompt configurations (e.g., rider on/off × reasoning_effort). Compares them on
token-F1 (primary) and reports per-cell winners.

Usage:
    python3 scripts/pick_winning_qa_prompt.py \
      --examples evaluation/results/musique_pilot_first10.json \
      --candidates label1=path1.json label2=path2.json ...
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    return 2 * (n / len(pt)) * (n / len(gt)) / ((n / len(pt)) + (n / len(gt)))


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def sub_em_score(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    return float(g in p) if g else 0.0


def score_run(path: Path, gold: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    d = json.loads(path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for cfg, info in d.get("runs", {}).items():
        rs = info.get("results") or []
        if not rs:
            continue
        f1s, ems, sems = [], [], []
        for r in rs:
            qid = r.get("question_id")
            pred = (
                r.get("short_answer")
                or r.get("final_answer_short")
                or r.get("answer")
                or r.get("final_answer", "")
            )
            if isinstance(pred, dict):
                pred = pred.get("answer", "")
            g = gold.get(qid, "")
            f1s.append(f1_score(str(pred), g))
            ems.append(em_score(str(pred), g))
            sems.append(sub_em_score(str(pred), g))
        n = len(rs)
        out[cfg] = {
            "n": n,
            "f1": sum(f1s) / n,
            "em": sum(ems) / n,
            "sub_em": sum(sems) / n,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="label=path entries (e.g., riderOn_low=eval/results/...)",
    )
    args = parser.parse_args()

    examples = json.loads(Path(args.examples).read_text())
    gold = {e["question_id"]: e.get("answer", "") for e in examples}

    # Score each candidate
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in args.candidates:
        if "=" not in entry:
            raise ValueError(f"expected label=path, got {entry}")
        label, path = entry.split("=", 1)
        if not Path(path).exists():
            print(f"  MISSING: {path}")
            continue
        results[label] = score_run(Path(path), gold)

    # Print per-config matrix and pick winner
    all_cfgs: List[str] = []
    for r in results.values():
        for cfg in r:
            if cfg not in all_cfgs:
                all_cfgs.append(cfg)

    print(f"\n=== Per-(QA config, prompt label) F1 matrix on n={len(gold)} dev ===")
    header = f"{'QA config':28s} | " + " | ".join(f"{lbl:14s}" for lbl in results.keys()) + " | winner"
    print(header)
    print("-" * len(header))
    winners: Dict[str, Tuple[str, float]] = {}
    for cfg in all_cfgs:
        row_vals: List[Tuple[str, float]] = []
        cells: List[str] = []
        for lbl, scored in results.items():
            s = scored.get(cfg)
            if s is None:
                cells.append(f"{'-':>14s}")
            else:
                cells.append(f"F1={s['f1']:.3f} EM={s['em']:.2f}".ljust(14))
                row_vals.append((lbl, s["f1"]))
        if row_vals:
            best = max(row_vals, key=lambda x: x[1])
            winners[cfg] = best
            cells.append(f"{best[0]} (F1={best[1]:.3f})")
        else:
            cells.append("none")
        print(f"{cfg:28s} | " + " | ".join(cells))

    print(f"\n=== Winning prompt per QA config ===")
    for cfg, (lbl, f1) in winners.items():
        print(f"  {cfg:28s} -> {lbl}  (F1={f1:.3f})")


if __name__ == "__main__":
    main()
