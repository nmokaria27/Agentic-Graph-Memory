from evaluation.governance.run_governance_benchmark import (
    build_examples,
    compute_metrics,
)
from evaluation.kgafe.baselines import OracleDomainWrapper
from evaluation.kgafe.benchmark_generator import BenchmarkQuestion
from evaluation.kgafe.evaluator import KGAFEEvaluator, EvaluationResult, KGAFEMetrics
from multi_agent_kg.core.domain_experts import Domain, OrgChart
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
    assert sub_questions[0]["target_domains"] == ["d1", "d2"]


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
