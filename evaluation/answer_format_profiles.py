"""Benchmark-specific answer-format profiles.

These live in the EVALUATION layer on purpose: the core QA system
(`multi_agent_kg.core`) never references a benchmark name. Harnesses pick a profile
here and inject the resulting generic :class:`AnswerFormatConfig` into the orchestrator.

Add a new benchmark by adding an entry to ``PROFILES``; nothing in core changes.
"""

from __future__ import annotations

from multi_agent_kg.core.config import AnswerFormatConfig


def _default() -> AnswerFormatConfig:
    # Mirrors the historical in-prompt wording (commit_mode from KGQA_COMMIT_MODE env).
    return AnswerFormatConfig()


def _locomo() -> AnswerFormatConfig:
    # LoComo is F1-scored over short conversational answers — keep prose terse.
    return AnswerFormatConfig(max_sentences=2, short_answer_words=5, style="minimal_span")


def _mab_substring() -> AnswerFormatConfig:
    # MemoryAgentBench primary metric is substring-EM: commit to an exact span and
    # allow a slightly longer window so the gold span is contained verbatim.
    return AnswerFormatConfig(
        max_sentences=2,
        short_answer_words=8,
        style="minimal_span",
        commit_mode=True,
    )


def _longmemeval() -> AnswerFormatConfig:
    # LongMemEval is LLM-judged "does the response contain the correct answer".
    # The answer must be stated explicitly (not implied); a short window keeps the
    # judge focused, but no commit_mode — knowledge-update answers often need one
    # qualifying clause ("as of the last conversation, X").
    return AnswerFormatConfig(
        max_sentences=3,
        short_answer_words=12,
        style="minimal_span",
        commit_mode=False,
    )


def _beam() -> AnswerFormatConfig:
    # BEAM uses LLM-as-judge with a rubric — answers are scored for semantic
    # compliance, not token F1. Allow fuller prose so the judge has enough
    # content to evaluate against each rubric criterion. Avoid bullet lists
    # which can fragment reasoning that the judge needs to read holistically.
    return AnswerFormatConfig(
        max_sentences=5,
        short_answer_words=30,
        style="verbose",
        commit_mode=False,
    )


# Factories (not instances) so each call re-reads env defaults and stays independent.
PROFILES = {
    "default": _default,
    "locomo": _locomo,
    "mab_substring": _mab_substring,
    "longmemeval": _longmemeval,
    "beam": _beam,
}


def get_answer_format(profile: str) -> AnswerFormatConfig:
    """Return a fresh AnswerFormatConfig for ``profile`` (KeyError-safe → default)."""
    factory = PROFILES.get((profile or "default").lower(), _default)
    return factory()


__all__ = ["PROFILES", "get_answer_format"]
