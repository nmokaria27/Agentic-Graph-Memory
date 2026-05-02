from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.governance import Domain
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.qa_orchestrator import DomainExpertAgent


def test_domain_expert_prompt_includes_domain_memory_evidence(monkeypatch) -> None:
    kg = KnowledgeGraph()
    kg.add_entity("bert", ["BERT"], "Method")
    kg.add_entity("classification", ["classification"], "Task")
    kg.add_triple(
        "bert",
        "Used-for",
        "classification",
        0.9,
        source="doc1",
        metadata={"evidence": "BERT is used for text classification in the abstract."},
    )
    domain = Domain(
        domain_id="methods",
        label="Methods",
        description="Methods and tasks",
        entity_ids={"bert", "classification"},
        relation_schema={"Used-for": "method usage"},
        topics=[],
    )
    domain.refresh_memory_card(kg)
    captured = {}

    def fake_chat_completion_json(*, messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return {
            "answer": "BERT is used for classification.",
            "coverage": 1.0,
            "evidence": ["(bert) -[Used-for]-> (classification)"],
            "confidence": 0.9,
        }

    monkeypatch.setattr(
        "multi_agent_kg.core.qa_orchestrator._chat_completion_json",
        fake_chat_completion_json,
    )

    expert = DomainExpertAgent(domain, kg, LLMConfig(model="test-model"))
    answer = expert.answer("What is BERT used for?")

    assert "DOMAIN MEMORY CARD" in captured["prompt"]
    assert "BERT is used for text classification in the abstract." in captured["prompt"]
    assert answer["answer"] == "BERT is used for classification."
