"""Hyperparameter search space for Stage-A retrieval tuning.

Kept separate from the optimizer so it is unit-testable without optuna and reused by
both tune_retrieval.py and the Stage-B re-tune. Ranges mirror the plan / Cognee paper.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Retrieval-time knobs only (no chunk_size — that needs a rebuild, handled in Stage B).
RETRIEVAL_MODES = ["hybrid", "graph_completion", "graph_summary", "chunk"]


def suggest(trial: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample one configuration from an optuna Trial.

    Returns ``(retrieval_kwargs, answer_format_kwargs)`` ready to splat into
    ``RetrievalConfig(**retrieval_kwargs)`` / ``AnswerFormatConfig(**format_kwargs)``.
    """
    retrieval = {
        "retrieval_mode": trial.suggest_categorical("retrieval_mode", RETRIEVAL_MODES),
        "entity_top_k": trial.suggest_int("entity_top_k", 3, 20),
        "triple_top_k": trial.suggest_int("triple_top_k", 10, 60),
        "domain_top_k": trial.suggest_int("domain_top_k", 2, 8),
        "entity_min_score": trial.suggest_float("entity_min_score", 0.40, 0.80),
        "seed_cap": trial.suggest_int("seed_cap", 10, 60),
        "focused_limit": trial.suggest_int("focused_limit", 20, 80),
        "max_hops": trial.suggest_int("max_hops", 2, 4),
    }
    answer_format = {
        "max_sentences": trial.suggest_int("max_sentences", 1, 4),
        "short_answer_words": trial.suggest_int("short_answer_words", 1, 10),
    }
    return retrieval, answer_format


def params_to_configs(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a flat best_params dict (from a finished study) back into the two kwarg sets."""
    fmt_keys = {"max_sentences", "short_answer_words"}
    retrieval = {k: v for k, v in params.items() if k not in fmt_keys}
    answer_format = {k: v for k, v in params.items() if k in fmt_keys}
    return retrieval, answer_format


__all__ = ["suggest", "params_to_configs", "RETRIEVAL_MODES"]
