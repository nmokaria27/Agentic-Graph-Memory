"""Stage B: coarse chunk_size grid (rebuilds KGs) + mandatory retrieval re-tune.

chunk_size changes graph topology, so the Stage-A retrieval optimum can shift. This
script:
  1. Loads Stage-A best retrieval/format params.
  2. For each chunk_size in the grid: rebuild the train-subset KGs at that size and
     score with the Stage-A config. Pick the best chunk_size.
  3. MANDATORY: run a shorter Stage-A TPE (~20 trials) on the winning chunk_size's
     rebuilt KGs to re-fit retrieval params to the new topology.

Only LoComo rebuilds are wired here (the first tune target).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

from scripts.tune.objective import SUBSET_DIR, evaluate
from scripts.tune.search_space import params_to_configs, suggest

DEFAULT_GRID = [400, 800, 1200, 1600, 2000]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-B chunk_size grid + re-tune")
    ap.add_argument("--benchmark", choices=["locomo"], default="locomo")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--cache-root", default="evaluation/results/tune/chunk_cache",
                    help="Per-chunk KG caches are written under here")
    ap.add_argument("--grid", default=",".join(map(str, DEFAULT_GRID)),
                    help="Comma-separated chunk sizes (chars)")
    ap.add_argument("--retune-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    best_path = SUBSET_DIR / f"{args.benchmark}_studyA_best.json"
    if not best_path.exists():
        raise SystemExit(f"Run tune_retrieval.py first — missing {best_path}")
    stageA = json.loads(best_path.read_text())
    retrieval_kwargs, format_kwargs = params_to_configs(stageA["best_params"])

    grid = [int(x) for x in args.grid.split(",")]
    cache_root = Path(args.cache_root)
    results = {}
    for cs in grid:
        cache = cache_root / f"cs_{cs}"
        score = evaluate(
            args.benchmark, retrieval_kwargs, format_kwargs,
            split="train", data_file=args.data_file,
            load_kg_dir=str(cache), save_kg_dir=str(cache), chunk_size=cs,
            model=args.model, embedding_model=args.embedding_model,
        )
        results[cs] = score
        print(f"chunk_size={cs}: {score:.4f}")

    best_cs = max(results, key=results.get)
    print(f"\nBest chunk_size: {best_cs} ({results[best_cs]:.4f})")

    # Mandatory re-tune on the winning topology's rebuilt KGs (now cached).
    cache = str(cache_root / f"cs_{best_cs}")

    def objective(trial: optuna.Trial) -> float:
        rk, fk = suggest(trial)
        return evaluate(
            args.benchmark, rk, fk, split="train", data_file=args.data_file,
            load_kg_dir=cache, model=args.model, embedding_model=args.embedding_model,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.retune_trials)
    print(f"Re-tuned best on chunk_size={best_cs}: {study.best_value:.4f}")

    out = SUBSET_DIR / f"{args.benchmark}_studyB_best.json"
    out.write_text(json.dumps({
        "grid_scores": results,
        "best_chunk_size": best_cs,
        "retuned_value": study.best_value,
        "retuned_params": study.best_params,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
