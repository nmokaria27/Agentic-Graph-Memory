"""Item ② — AnswerFormatConfig sizing + rider gating, and the eval-only profiles.

Core must stay generic: the only place benchmark names appear is the eval profile
table. These tests assert the rider/length knobs respond to config, and that the
default profile reproduces historical wording.
"""

from multi_agent_kg.core._qa_commit import build_rider
from multi_agent_kg.core.config import AnswerFormatConfig
from evaluation.answer_format_profiles import PROFILES, get_answer_format


def test_default_config_matches_history() -> None:
    f = AnswerFormatConfig()
    assert f.max_sentences == 3
    assert f.short_answer_words == 5
    assert f.style == "minimal_span"


def test_rider_is_gated_and_sized() -> None:
    assert build_rider(AnswerFormatConfig(commit_mode=False)) == ""
    rider = build_rider(AnswerFormatConfig(commit_mode=True, short_answer_words=8))
    assert "1-8 words" in rider
    assert "DO NOT HEDGE" in rider


def test_style_changes_short_answer_instruction() -> None:
    # `style` must actually alter prompt wording, not be inert.
    minimal = AnswerFormatConfig(style="minimal_span", short_answer_words=8)
    dry = AnswerFormatConfig(style="short_dry", short_answer_words=8)
    verbose = AnswerFormatConfig(style="verbose", short_answer_words=8)
    i_min = minimal.short_answer_instruction()
    i_dry = dry.short_answer_instruction()
    i_verb = verbose.short_answer_instruction()
    assert i_min != i_dry != i_verb and i_min != i_verb
    assert "1-8 words" in i_min and "1-8 words" in i_dry
    assert "MINIMAL" in i_min and "dry" in i_dry and "sentence is acceptable" in i_verb


def test_invalid_style_rejected() -> None:
    try:
        AnswerFormatConfig(style="rambling")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid style should raise")


def test_eval_profiles_are_distinct_and_generic() -> None:
    # Profiles live only in the eval layer; each returns a fresh generic object.
    assert set(PROFILES) >= {"default", "locomo", "mab_substring"}
    locomo = get_answer_format("locomo")
    mab = get_answer_format("mab_substring")
    assert locomo.max_sentences == 2
    assert mab.commit_mode is True and mab.short_answer_words == 8
    # Unknown profile falls back to default rather than raising.
    assert get_answer_format("does_not_exist").max_sentences == 3
