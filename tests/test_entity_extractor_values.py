from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.base import AgentContext
from multi_agent_kg.core import KnowledgeGraph, LLMConfig


def test_open_domain_entity_prompt_includes_answer_bearing_values(monkeypatch) -> None:
    extractor = EntityExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
        enable_deterministic_value_harvesting=True,
    )
    captured = {}

    def fake_call_llm(*, prompt, **kwargs):
        captured["prompt"] = prompt
        return {"entities": []}

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)

    extractor._extract_entities_combined(
        text="The arena seats 3,677 people and opened in 1999.",
        entity_types=["PERSON", "ORGANIZATION"],
        domain="general",
        strict_types=False,
    )

    assert "OPEN-DOMAIN VALUE ENTITY RULES" in captured["prompt"]
    assert "3,677 seated" in captured["prompt"]
    assert "1999" in captured["prompt"]


def test_fixed_schema_entity_prompt_does_not_add_open_domain_value_rules(monkeypatch) -> None:
    extractor = EntityExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
        enable_deterministic_value_harvesting=True,
    )
    captured = {}

    def fake_call_llm(*, prompt, **kwargs):
        captured["prompt"] = prompt
        return {"entities": []}

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)

    extractor._extract_entities_combined(
        text="CNN is used for classification.",
        entity_types=["Method", "Task"],
        domain="FixedSchema",
        strict_types=True,
    )

    assert "OPEN-DOMAIN VALUE ENTITY RULES" not in captured["prompt"]


def test_open_domain_entity_run_adds_answer_bearing_values(monkeypatch) -> None:
    extractor = EntityExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
        enable_deterministic_value_harvesting=True,
    )

    monkeypatch.setattr(extractor, "_extract_entities_combined", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        extractor,
        "call_llm",
        lambda *args, **kwargs: {
            "entity_groups": [
                {
                    "canonical_id": "3677_seated",
                    "canonical_name": "3,677 seated",
                    "mentions": ["3,677 seated"],
                    "type": "Quantity",
                    "is_known_entity": False,
                },
                {
                    "canonical_id": "from_1986_to_2013",
                    "canonical_name": "from 1986 to 2013",
                    "mentions": ["from 1986 to 2013"],
                    "type": "DateRange",
                    "is_known_entity": False,
                },
            ]
        },
    )

    result = extractor.run(
        AgentContext(
            document_id="doc1",
            text="The arena has 3,677 seated capacity. Ferguson managed Manchester United from 1986 to 2013.",
        ),
        segments=[
            {
                "segment_id": "doc1_s0",
                "text": "The arena has 3,677 seated capacity. Ferguson managed Manchester United from 1986 to 2013.",
            }
        ],
        domain_config={"entity_types": [{"type": "Entity"}]},
    )

    labels = {item["text"] for item in result.items}
    assert "3,677 seated" in labels
    assert "from 1986 to 2013" in labels
    assert all(item.get("source_text") for item in result.items)


def test_open_domain_entity_run_adds_dates_and_measurements(monkeypatch) -> None:
    extractor = EntityExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
        enable_deterministic_value_harvesting=True,
    )

    monkeypatch.setattr(extractor, "_extract_entities_combined", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        extractor,
        "call_llm",
        lambda *args, **kwargs: {"entity_groups": []},
    )

    result = extractor.run(
        AgentContext(
            document_id="doc1",
            text="Mike Medavoy was born January 21, 1941. Ofu Airport is located one mile southeast of Ofu.",
        ),
        segments=[
            {
                "segment_id": "doc1_s0",
                "text": "Mike Medavoy was born January 21, 1941. Ofu Airport is located one mile southeast of Ofu.",
            }
        ],
        domain_config={"entity_types": [{"type": "Entity"}]},
    )

    by_text = {item["text"]: item for item in result.items}
    assert "January 21, 1941" in by_text
    assert by_text["January 21, 1941"]["type"] == "Date"
    assert "one mile" not in by_text  # words are left to the LLM; deterministic pass handles numeric values.


def test_coreference_preserves_source_segments_through_public_run(monkeypatch) -> None:
    extractor = EntityExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
    )

    extracted_by_text = {
        "CNN is used for classification.": [
            {"id": "cnn", "text": "CNN", "labels": ["CNN"], "type": "Method", "confidence": 0.9}
        ],
        "The neural network improves classification.": [
            {
                "id": "neural_network",
                "text": "neural network",
                "labels": ["neural network"],
                "type": "Method",
                "confidence": 0.9,
            }
        ],
    }

    def fake_extract(text, *args, **kwargs):
        return extracted_by_text[text]

    def fake_call_llm(*args, **kwargs):
        return {
            "entity_groups": [
                {
                    "canonical_id": "cnn",
                    "canonical_name": "CNN",
                    "mentions": ["CNN", "neural network"],
                    "type": "Method",
                    "is_known_entity": False,
                }
            ]
        }

    monkeypatch.setattr(extractor, "_extract_entities_combined", fake_extract)
    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)

    result = extractor.run(
        AgentContext(document_id="doc1", text="CNN is used for classification. The neural network improves classification."),
        segments=[
            {"segment_id": "doc1_s0", "text": "CNN is used for classification."},
            {"segment_id": "doc1_s1", "text": "The neural network improves classification."},
        ],
        domain_config={
            "entity_types": [{"type": "Method"}],
            "confidence": 0.95,
            "reasoning": "Fixed schema",
        },
    )

    assert len(result.items) == 1
    resolved = result.items[0]
    assert resolved["source_segment"] == "doc1_s0"
    assert resolved["source_segments"] == ["doc1_s0", "doc1_s1"]
    assert resolved["source_document_id"] == "doc1"
