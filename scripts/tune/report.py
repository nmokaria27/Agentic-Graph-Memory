"""Summarize tuning results and emit the winning config as JSON.

IMPORTANT: the winning config is written ONLY to
``evaluation/results/tune/<bench>_best_config.json``. It is deliberately NOT baked into
RetrievalConfig defaults — LoComo/MAB-optimized values would regress SciERC/HotPotQA.
Eval scripts load this JSON explicitly when they want the tuned config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tune.objective import SUBSET_DIR
from scripts.tune.search_space import params_to_configs


def main() -> None:
    ap = argparse.ArgumentParser(description="Report tuning results")
    ap.add_argument("--benchmark", choices=["locomo", "mab"], required=True)
    args = ap.parse_args()

    stageA_path = SUBSET_DIR / f"{args.benchmark}_studyA_best.json"
    stageB_path = SUBSET_DIR / f"{args.benchmark}_studyB_best.json"
    if not stageA_path.exists():
        raise SystemExit(f"Missing {stageA_path}; run tune_retrieval.py first")

    stageA = json.loads(stageA_path.read_text())
    best_params = dict(stageA["best_params"])
    best_value = stageA["best_value"]
    best_chunk_size = None

    if stageB_path.exists():
        stageB = json.loads(stageB_path.read_text())
        # Stage B re-tune supersedes Stage A retrieval params on the winning topology.
        if stageB.get("retuned_value", -1) >= best_value:
            best_params = dict(stageB["retuned_params"])
            best_value = stageB["retuned_value"]
        best_chunk_size = stageB.get("best_chunk_size")

    retrieval_kwargs, format_kwargs = params_to_configs(best_params)
    payload = {
        "benchmark": args.benchmark,
        "score": best_value,
        "chunk_size": best_chunk_size,
        "retrieval_config": retrieval_kwargs,
        "answer_format_config": format_kwargs,
    }
    out = SUBSET_DIR / f"{args.benchmark}_best_config.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"=== {args.benchmark} tuned config (score {best_value:.4f}) ===")
    print(f"chunk_size: {best_chunk_size}")
    print(f"RetrievalConfig(**{retrieval_kwargs})")
    print(f"AnswerFormatConfig(**{format_kwargs})")
    print(f"\nWrote {out} (NOT applied to config.py defaults — load it explicitly).")


if __name__ == "__main__":
    main()
