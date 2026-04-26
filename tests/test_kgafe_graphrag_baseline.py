from evaluation.kgafe.baselines import GraphRAGQAOrchestrator, build_baseline_system
from multi_agent_kg.core import LLMConfig
from multi_agent_kg.core.domain_experts import Domain, OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def _sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity("bert", ["BERT"], "Method")
    kg.add_entity("classification", ["classification"], "Task")
    kg.add_entity("glue", ["GLUE"], "Dataset")
    kg.add_entity("accuracy", ["accuracy"], "Metric")
    kg.add_triple("bert", "Used-for", "classification", 0.9)
    kg.add_triple("glue", "Evaluate-for", "bert", 0.8)
    kg.add_triple("accuracy", "Evaluate-for", "bert", 0.8)
    return kg


def test_build_baseline_system_constructs_graphrag() -> None:
    kg = _sample_kg()
    org_chart = OrgChart(
        domains=[
            Domain(
                domain_id="global",
                label="Global",
                description="All entities",
                entity_ids=list(kg.entities.keys()),
                relation_schema=["Used-for", "Evaluate-for"],
                topics=[],
            )
        ],
        cross_domain_relations=[],
    )

    result = build_baseline_system(
        "graphrag_basic",
        kg=kg,
        llm_config=LLMConfig(model="test-model"),
        org_chart=org_chart,
        advanced_orchestrator_cls=object,
    )

    assert isinstance(result.qa_system, GraphRAGQAOrchestrator)
    assert len(result.qa_system.communities) >= 1


def test_graphrag_query_returns_final_answer(monkeypatch) -> None:
    kg = _sample_kg()
    system = GraphRAGQAOrchestrator(kg, LLMConfig(model="test-model"))

    def fake_chat_completion_json(*args, **kwargs):
        return {
            "answer": "BERT is used for classification.",
            "coverage": 0.8,
            "evidence": ["(bert) -[Used-for]-> (classification)"],
            "confidence": 0.9,
            "out_of_scope_aspects": [],
        }

    monkeypatch.setattr(
        "evaluation.kgafe.baselines.chat_completion_json",
        fake_chat_completion_json,
    )

    result = system.query("What is BERT used for?")

    assert result["final_answer"] == "BERT is used for classification."
    assert result["retrieval_mode"] == "community_summary"
    assert result["community_ids"]
