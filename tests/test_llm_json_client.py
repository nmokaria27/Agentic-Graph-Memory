from multi_agent_kg.llm.openai_client import _unwrap_json_mode_array


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
