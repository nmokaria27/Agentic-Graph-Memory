from evaluation.governance.run_governance_benchmark import (
    build_examples,
    compute_metrics,
)
from evaluation.kgafe.baselines import OracleDomainWrapper
from evaluation.kgafe.benchmark_generator import BenchmarkQuestion
from evaluation.kgafe.evaluator import KGAFEEvaluator, EvaluationResult, KGAFEMetrics
from evaluation.kgafe.export_human_eval import build_question_rows
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.domain_experts import Domain, OrgChart, QAOrchestrator
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def test_evaluator_prefers_query_benchmark_question(monkeypatch) -> None:
    kg = KnowledgeGraph()
    evaluator = KGAFEEvaluator(kg=kg, enable_judge_panel=False)

    question = BenchmarkQuestion(
        question_id="q1",
        question="Q1",
        gold_answer="G1",
        question_type="negative",
        difficulty="easy",
    )

    class StubQA:
        def __init__(self):
            self.called = False

        def query_benchmark_question(self, bq):
            self.called = True
            return {"final_answer": f"oracle for {bq.question}"}

        def query(self, question):
            raise AssertionError("query() should not be called when query_benchmark_question exists")

    def fake_evaluate_answer(question, answer, gold_answer=None, relevant_triples=None):
        return EvaluationResult(
            question=question,
            answer=answer,
            gold_answer=gold_answer,
            metrics=KGAFEMetrics(kgafe_score=0.5),
        )

    qa = StubQA()
    monkeypatch.setattr(evaluator, "evaluate_answer", fake_evaluate_answer)
    result = evaluator.evaluate_benchmark_questions([question], qa_system=qa)

    assert qa.called is True
    assert result.individual_results[0].answer == "oracle for Q1"


def test_oracle_domain_wrapper_forces_expected_domains() -> None:
    seen = {}

    class StubSystem:
        def _decompose_and_route(self, question):
            return {}

        def query(self, question):
            seen["routing"] = self._decompose_and_route(question)
            return {"final_answer": "ok"}

    wrapper = OracleDomainWrapper(StubSystem())
    wrapper.qa_system.max_routed_domains = 4
    bq = BenchmarkQuestion(
        question_id="q1",
        question="How does A connect to B?",
        gold_answer="",
        question_type="cross_domain",
        difficulty="hard",
        expected_domains=["d1", "d2", "d3"],
    )
    wrapper.query_benchmark_question(bq)

    sub_questions = seen["routing"]["sub_questions"]
    assert sub_questions[0]["target_domains"] == ["d1", "d2", "d3"]


def test_basic_routing_normalization_keeps_up_to_four_domains(monkeypatch) -> None:
    kg = KnowledgeGraph()
    for entity_id in ["a", "b", "c", "d"]:
        kg.add_entity(entity_id, [entity_id.upper()], "TYPE")

    org_chart = OrgChart(
        domains=[
            Domain(domain_id="d1", label="D1", description="", entity_ids={"a"}),
            Domain(domain_id="d2", label="D2", description="", entity_ids={"b"}),
            Domain(domain_id="d3", label="D3", description="", entity_ids={"c"}),
            Domain(domain_id="d4", label="D4", description="", entity_ids={"d"}),
        ],
        cross_domain_relations=[],
    )

    orchestrator = QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=LLMConfig(model="test"))

    def fake_chat_completion_json(*args, **kwargs):
        return {
            "sub_questions": [
                {
                    "question": "Q",
                    "target_domains": ["d1", "d2", "d3", "d4"],
                    "context": "",
                }
            ]
        }

    monkeypatch.setattr("multi_agent_kg.core.domain_experts.chat_completion_json", fake_chat_completion_json)
    routed = orchestrator._decompose_and_route("Q")
    assert routed["sub_questions"][0]["target_domains"] == ["d1", "d2", "d3", "d4"]


def test_basic_query_adds_ownership_hints_and_global_fallback() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("entity_a", ["Entity A"], "TYPE")
    kg.add_entity("entity_b", ["Entity B"], "TYPE")
    kg.add_triple("entity_a", "RELATES_TO", "entity_b", 1.0)

    org_chart = OrgChart(
        domains=[
            Domain(
                domain_id="d1",
                label="D1",
                description="",
                entity_ids={"entity_a"},
                relation_schema={"RELATES_TO": "rel"},
            ),
            Domain(
                domain_id="d2",
                label="D2",
                description="",
                entity_ids=set(),
                relation_schema={},
            ),
        ],
        cross_domain_relations=[],
    )
    orchestrator = QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=LLMConfig(model="test"))

    seen = {}

    class StubExpert:
        def answer(self, query, context=""):
            seen["context"] = context
            return {
                "domain_id": "d1",
                "answer": "",
                "coverage": 0.0,
                "confidence": 0.0,
                "evidence": [],
                "out_of_scope_aspects": ["entity_b"],
            }

    class StubFallback:
        def answer(self, query, context=""):
            seen["fallback_context"] = context
            return {
                "domain_id": "global_fallback",
                "answer": "entity_a relates to entity_b.",
                "coverage": 0.8,
                "confidence": 0.7,
                "evidence": ["(entity_a) -[RELATES_TO]-> (entity_b)"],
                "out_of_scope_aspects": [],
            }

    class StubIrrelevantExpert:
        def answer(self, query, context=""):
            return {
                "domain_id": "d2",
                "answer": "",
                "coverage": 0.0,
                "confidence": 0.0,
                "evidence": [],
                "out_of_scope_aspects": ["entity_a", "entity_b"],
            }

    orchestrator.experts = {"d1": StubExpert(), "d2": StubIrrelevantExpert()}
    orchestrator.global_fallback_expert = StubFallback()
    orchestrator._decompose_and_route = lambda question: {
        "sub_questions": [
            {
                "question": question,
                "target_domains": ["d1"],
                "context": "",
            }
        ]
    }
    orchestrator._synthesize = lambda *args, **kwargs: {
        "answer": "ok",
        "coverage": 1.0,
        "confidence": 1.0,
        "gaps": [],
    }

    result = orchestrator.query("How does Entity A relate to Entity B?")

    assert "Ownership hints:" in seen["context"]
    assert "entity_b: currently unowned in the domain chart" in seen["context"]
    assert "Unowned query entities: entity_b" in seen["fallback_context"]
    assert any(resp["domain_id"] == "global_fallback" for resp in result["domain_responses"])


def test_advanced_routing_normalization_keeps_up_to_four_domains(monkeypatch) -> None:
    kg = KnowledgeGraph()
    for entity_id in ["a", "b", "c", "d"]:
        kg.add_entity(entity_id, [entity_id.upper()], "TYPE")

    org_chart = OrgChart(
        domains=[
            Domain(domain_id="d1", label="D1", description="", entity_ids={"a"}),
            Domain(domain_id="d2", label="D2", description="", entity_ids={"b"}),
            Domain(domain_id="d3", label="D3", description="", entity_ids={"c"}),
            Domain(domain_id="d4", label="D4", description="", entity_ids={"d"}),
        ],
        cross_domain_relations=[],
    )

    orchestrator = AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=kg,
        llm_config=LLMConfig(model="test"),
        max_exploration_rounds=1,
        enable_debate=False,
        enable_critic=False,
    )

    def fake_chat_completion_json(*args, **kwargs):
        return {
            "sub_questions": [
                {
                    "question": "Q",
                    "target_domains": ["d1", "d2", "d3", "d4"],
                    "context": "",
                }
            ]
        }

    monkeypatch.setattr("multi_agent_kg.core.advanced_qa.chat_completion_json", fake_chat_completion_json)
    routed = orchestrator._decompose_and_route("Q", "")
    assert routed["sub_questions"][0]["target_domains"] == ["d1", "d2", "d3", "d4"]


def test_human_eval_export_blinds_answer_order() -> None:
    data = {
        "runs": {
            "flat_basic": {
                "individual_results": [
                    {"question_id": "q1", "question": "Q1", "question_type": "comparison", "difficulty": "hard", "answer": "flat", "gold_answer": "gold"},
                    {"question_id": "q2", "question": "Q2", "question_type": "negative", "difficulty": "easy", "answer": "flat2", "gold_answer": "gold2"},
                ]
            },
            "domain_basic": {
                "individual_results": [
                    {"question_id": "q1", "question": "Q1", "question_type": "comparison", "difficulty": "hard", "answer": "domain", "gold_answer": "gold"},
                    {"question_id": "q2", "question": "Q2", "question_type": "negative", "difficulty": "easy", "answer": "domain2", "gold_answer": "gold2"},
                ]
            },
            "domain_advanced": {
                "individual_results": [
                    {"question_id": "q1", "question": "Q1", "question_type": "comparison", "difficulty": "hard", "answer": "adv", "gold_answer": "gold"},
                    {"question_id": "q2", "question": "Q2", "question_type": "negative", "difficulty": "easy", "answer": "adv2", "gold_answer": "gold2"},
                ]
            },
        }
    }

    rows = build_question_rows(
        data=data,
        configs=["flat_basic", "domain_basic", "domain_advanced"],
        sample_size=2,
        seed=7,
    )

    assert len(rows) == 2
    assert set(rows[0]["_answer_key"].values()) == {
        "flat_basic",
        "domain_basic",
        "domain_advanced",
    }


def test_governance_benchmark_builds_positive_and_negative_examples() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("a", ["A"], "TYPE")
    kg.add_entity("b", ["B"], "TYPE")
    kg.add_entity("c", ["C"], "TYPE")
    kg.add_triple("a", "REL", "b", 1.0)
    kg.add_triple("b", "REL", "c", 1.0)

    d1 = Domain(domain_id="d1", label="D1", description="", entity_ids={"a"}, relation_schema={}, topics=[])
    d2 = Domain(domain_id="d2", label="D2", description="", entity_ids={"b", "c"}, relation_schema={}, topics=[])
    org_chart = OrgChart(domains=[d1, d2], cross_domain_relations=kg.triples)

    examples = build_examples(kg, org_chart, num_positive=1, num_negative=1, seed=7)

    assert len(examples) == 2
    labels = sorted(example.label for example in examples)
    assert labels == ["negative", "positive"]


def test_governance_benchmark_metrics() -> None:
    rows = [
        {
            "label": "positive",
            "expected_domains": ["d1"],
            "predicted_domains": ["d1"],
            "expected_assignment_type": "single_owner",
            "predicted_assignment_type": "single_owner",
            "decision": "approve",
        },
        {
            "label": "negative",
            "expected_domains": ["d1", "d2"],
            "predicted_domains": ["d1"],
            "expected_assignment_type": "cross_domain",
            "predicted_assignment_type": "single_owner",
            "decision": "reject",
        },
    ]

    metrics = compute_metrics(rows)
    assert metrics["routing_exact_match"] == 0.5
    assert metrics["assignment_type_accuracy"] == 0.5
    assert metrics["accept_recall_positive"] == 1.0
    assert metrics["reject_recall_negative"] == 1.0
