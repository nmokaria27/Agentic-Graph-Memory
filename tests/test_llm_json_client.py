import multi_agent_kg.llm.openai_client as oc
from multi_agent_kg.llm.openai_client import (
    _extract_json,
    _model_is_thinking,
    _unwrap_json_mode_array,
    chat_completion_typed,
)
from multi_agent_kg.schemas_llm import EntityExtractionOut


def test_deepseek_v4_detected_as_thinking() -> None:
    # v4-flash emits CoT prose before JSON: must skip json_object mode so the
    # CoT-tolerant _extract_json parser runs instead of crashing on prose.
    assert _model_is_thinking("accounts/fireworks/models/deepseek-v4-flash")
    assert _model_is_thinking("deepseek-v4")
    # Non-thinking instruct models stay on constrained JSON decoding.
    assert not _model_is_thinking("accounts/fireworks/models/deepseek-v3")


def test_reasoning_model_families_detected_as_thinking() -> None:
    # Fireworks reasoning families that emit CoT — must skip json_object mode.
    for model in (
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/gpt-oss-20b",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
        "accounts/fireworks/models/qwen3p7-plus",
    ):
        assert _model_is_thinking(model), model
    # Verified-working / non-thinking instruct models stay on json_object mode.
    for model in (
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/deepseek-v3",
    ):
        assert not _model_is_thinking(model), model


def test_extract_json_digs_json_out_of_cot_prose() -> None:
    # Mirrors the v4-flash failure mode: reasoning prose, then the JSON object.
    text = (
        "We are given a conversation. We need to extract entities. Let's go.\n"
        '{"entities": [{"text": "Caroline", "type": "PERSON"}]}'
    )
    assert _extract_json(text) == {"entities": [{"text": "Caroline", "type": "PERSON"}]}


def test_json_mode_array_wrapper_unwraps_single_list_value() -> None:
    assert _unwrap_json_mode_array({"domains": [{"domain_id": "people"}]}) == [
        {"domain_id": "people"}
    ]


def test_json_mode_array_wrapper_keeps_regular_objects() -> None:
    payload = {"answer": "x", "evidence": []}
    assert _unwrap_json_mode_array(payload) == payload


def test_json_mode_array_wrapper_keeps_multi_key_objects() -> None:
    payload = {"domains": [], "metadata": {"source": "test"}}
    assert _unwrap_json_mode_array(payload) == payload


def _stub_responses(monkeypatch, responses):
    """Patch chat_completion_json to return queued responses in order."""
    queue = list(responses)
    calls = {"n": 0}

    def fake(messages, **kwargs):
        calls["n"] += 1
        return queue.pop(0)

    monkeypatch.setattr(oc, "chat_completion_json", fake)
    return calls


def test_typed_validates_valid_response(monkeypatch) -> None:
    _stub_responses(monkeypatch, [
        {"entities": [{"text": "Caroline", "type": "PERSON", "confidence": "0.8"}]},
    ])
    out = chat_completion_typed([{"role": "user", "content": "x"}], EntityExtractionOut)
    assert isinstance(out, EntityExtractionOut)
    assert out.entities[0].text == "Caroline"
    assert out.entities[0].confidence == 0.8


def test_typed_reprompts_then_succeeds(monkeypatch) -> None:
    # First response wrong shape (entities is a str -> ValidationError), then valid.
    calls = _stub_responses(monkeypatch, [
        {"entities": "not-a-list-or-dict-item"},
        {"entities": [{"text": "Insulin"}]},
    ])
    out = chat_completion_typed([{"role": "user", "content": "x"}], EntityExtractionOut)
    assert calls["n"] == 2
    assert [e.text for e in out.entities] == ["Insulin"]


def test_typed_exhausts_retries_returns_empty(monkeypatch) -> None:
    bad = {"entities": "still-bad"}
    calls = _stub_responses(monkeypatch, [bad, bad, bad])
    out = chat_completion_typed(
        [{"role": "user", "content": "x"}],
        EntityExtractionOut,
        max_validation_retries=2,
    )
    assert calls["n"] == 3  # initial + 2 retries
    assert isinstance(out, EntityExtractionOut)
    assert out.entities == []
