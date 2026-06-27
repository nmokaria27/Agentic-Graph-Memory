from multi_agent_kg.llm.openai_client import (
    _extract_json,
    _model_is_thinking,
    _unwrap_json_mode_array,
)


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
