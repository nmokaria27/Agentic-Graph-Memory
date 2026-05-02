"""Re-score MuSiQue/Hotpot QA outputs with a substring-EM secondary metric.

Token-F1 already exists; substring-EM is added uniformly so verbose grounded
answers (e.g., GraphRAG's `... located in Randall County, Texas [Data: Sources]`)
are not penalized for verbosity when the gold span is contained.

Substring-EM = 1.0 if the SQuAD-normalized gold answer is a substring of the
SQuAD-normalized prediction; else 0.0.

Handles both file shapes:
- Flat:  {"results": [...]}
- Runs:  {"runs": {name: {"results": [...]}}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import normalize_answer


def _substring_em(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction or "")
    g = normalize_answer(gold or "")
    if not g:
        return 0.0
    if not pred:
        return 0.0
    return 1.0 if g in pred else 0.0


def _pick_pred(row: Dict[str, Any]) -> str:
    for key in ("answer", "raw_answer", "short_answer", "scored_answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _rescore_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    sub_em_sum = 0.0
    f1_sum = 0.0
    em_sum = 0.0
    answered = 0
    for row in rows:
        pred = _pick_pred(row)
        gold = row.get("gold_answer", "")
        sub_em = _substring_em(pred, gold)
        m = row.setdefault("metrics", {})
        m["substring_em"] = sub_em
        sub_em_sum += sub_em
        f1_sum += float(m.get("token_f1", 0.0))
        em_sum += float(m.get("exact_match", 0.0))
        if pred.strip():
            answered += 1
    return {
        "num_questions": n,
        "exact_match": round(em_sum / n, 4) if n else 0.0,
        "substring_em": round(sub_em_sum / n, 4) if n else 0.0,
        "token_f1": round(f1_sum / n, 4) if n else 0.0,
        "answer_rate": round(answered / n, 4) if n else 0.0,
    }


def _process(data: Dict[str, Any]) -> Dict[str, Any]:
    if "runs" in data and isinstance(data["runs"], dict):
        for name, run in data["runs"].items():
            rows = run.get("results", [])
            run["aggregate_metrics"] = _rescore_rows(rows)
    elif "results" in data and isinstance(data["results"], list):
        data["aggregate_metrics"] = _rescore_rows(data["results"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--suffix", default="_subem")
    args = parser.parse_args()

    summary: Dict[str, Any] = {}
    for path_str in args.inputs:
        path = Path(path_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        data = _process(data)
        out = path.with_name(path.stem + args.suffix + path.suffix)
        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        if "runs" in data:
            for name, run in data["runs"].items():
                summary[f"{path.stem}::{name}"] = run["aggregate_metrics"]
        else:
            summary[path.stem] = data.get("aggregate_metrics", {})
        print(f"wrote {out}", flush=True)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
