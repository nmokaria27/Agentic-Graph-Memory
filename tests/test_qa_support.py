from evaluation.qa_support import aggregate_support_results, evaluate_question_support
from multi_agent_kg.core.governance import Domain, OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def test_answer_support_detects_entity_evidence_and_domain_memory() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("scott_derrickson", ["Scott Derrickson"], "PERSON")
    kg.add_entity("american", ["American"], "NATIONALITY")
    kg.add_triple(
        "scott_derrickson",
        "Has-nationality",
        "american",
        0.9,
        metadata={"evidence": "Scott Derrickson is an American director."},
    )
    domain = Domain(
        domain_id="people",
        label="People",
        description="People and biographical facts",
        entity_ids={"scott_derrickson", "american"},
        relation_schema={},
    )
    org = OrgChart(domains=[domain], cross_domain_relations=[])
    org.refresh_memory_cards(kg)

    result = evaluate_question_support(
        kg=kg,
        org_chart=org,
        question_id="q1",
        question="What nationality is Scott Derrickson?",
        gold_answer="American",
    )

    assert result.answer_supported
    assert result.answer_in_entity
    assert result.answer_in_triple
    assert result.answer_in_evidence
    assert result.answer_in_domain_memory
    assert result.likely_failure_stage == "retrieval_or_synthesis_gap"


def test_answer_support_classifies_missing_answer_as_extraction_miss() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("scott_derrickson", ["Scott Derrickson"], "PERSON")

    result = evaluate_question_support(
        kg=kg,
        question_id="q1",
        question="What nationality is Scott Derrickson?",
        gold_answer="American",
    )

    assert not result.answer_supported
    assert result.likely_failure_stage == "extraction_miss"
    assert aggregate_support_results([result])["answer_support_rate"] == 0.0
