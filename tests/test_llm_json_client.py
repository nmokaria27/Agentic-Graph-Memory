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
