"""Stage A: retrieval-only hyperparameter tuning with Optuna TPE.

Sweeps RetrievalConfig + AnswerFormatConfig knobs against pre-built KGs
(``--load-kg-dir``) so each trial is a cheap rescore (no graph rebuild). Optimizes
LoComo overall_f1 or MAB accuracy on the frozen train subset; then reports the best
config's score on the held-out test subset (generalization check).

The winning config is written to evaluation/results/tune/<bench>_best_config.json by
report.py — it is NOT baked into RetrievalConfig defaults.

Example:
  python -m scripts.tune.tune_retrieval --benchmark locomo \
      --data-file evaluation/LoComo/data/locomo10.json \
      --load-kg-dir evaluation/results/locomo_kg_cache \
      --model accounts/fireworks/models/deepseek-v3 --n-trials 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

from scripts.tune.objective import SUBSET_DIR, evaluate
from scripts.tune.search_space import params_to_configs, suggest


def make_objective(args):
    def objective(trial: optuna.Trial) -> float:
        retrieval_kwargs, format_kwargs = suggest(trial)
        return evaluate(
            args.benchmark,
            retrieval_kwargs,
            format_kwargs,
            split="train",
            load_kg_dir=args.load_kg_dir,
            data_file=args.data_file,
            model=args.model,
            embedding_model=args.embedding_model,
            dataset=args.dataset,
            max_samples=args.max_samples,
        )

    return objective


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-A retrieval tuning (Optuna TPE)")
    ap.add_argument("--benchmark", choices=["locomo", "mab"], required=True)
    ap.add_argument("--data-file", default=None, help="locomo10.json (LoComo only)")
    ap.add_argument("--load-kg-dir", required=True, help="Pre-built KG cache (rescore-only)")
    ap.add_argument("--model", default=None, help="Cheap model recommended (e.g. DeepSeek V3)")
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--dataset", default="eventqa", help="MAB dataset name")
    ap.add_argument("--max-samples", type=int, default=24, help="MAB examples per trial")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"{args.benchmark}_studyA"
    storage = f"sqlite:///{SUBSET_DIR / (study_name + '.db')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )
    study.optimize(make_objective(args), n_trials=args.n_trials)

    print(f"\nBest {args.benchmark} train score: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Generalization check on the held-out test subset.
    retrieval_kwargs, format_kwargs = params_to_configs(study.best_params)
    try:
        test_score = evaluate(
            args.benchmark, retrieval_kwargs, format_kwargs,
            split="test", load_kg_dir=args.load_kg_dir, data_file=args.data_file,
            model=args.model, embedding_model=args.embedding_model,
            dataset=args.dataset, max_samples=args.max_samples,
        )
        print(f"Held-out test score: {test_score:.4f}")
    except Exception as exc:  # held-out subset may be absent in a smoke run
        print(f"(held-out eval skipped: {exc})")

    out = SUBSET_DIR / f"{args.benchmark}_studyA_best.json"
    out.write_text(json.dumps(
        {"best_value": study.best_value, "best_params": study.best_params}, indent=2
    ))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
