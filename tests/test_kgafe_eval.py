from evaluation.kgafe.atomic_decomposer import AtomicFact
from evaluation.kgafe.benchmark_generator import BenchmarkGenerator, BenchmarkQuestion
from evaluation.kgafe.evaluator import EvaluationResult, KGAFEEvaluator, KGAFEMetrics
from evaluation.kgafe.triple_verifier import VerificationResult, Verdict
from evaluation.kgafe.triple_verifier import TripleVerifier
from multi_agent_kg.core.domain_experts import Domain, OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def test_compute_metrics_distinguishes_faithfulness_and_precision() -> None:
    evaluator = KGAFEEvaluator(kg=KnowledgeGraph(), enable_judge_panel=False)
    facts = [
        AtomicFact("f1", "a", "a"),
        AtomicFact("f2", "b", "b"),
        AtomicFact("f3", "c", "c"),
        AtomicFact("f4", "d", "d"),
        AtomicFact("f5", "e", "e"),
    ]
    verifications = [
        VerificationResult("f1", Verdict.SUPPORTED, 1, 1.0),
        VerificationResult("f2", Verdict.SUPPORTED, 1, 1.0),
        VerificationResult("f3", Verdict.PARTIALLY_SUPPORTED, 3, 0.6),
        VerificationResult("f4", Verdict.CONTRADICTED, 1, 1.0),
        VerificationResult("f5", Verdict.UNVERIFIABLE, 3, 0.2),
    ]

    metrics = evaluator._compute_metrics(
        facts=facts,
        verifications=verifications,
        judge_verdict=None,
        question="q",
        answer="a",
        relevant_triples=None,
    )

    assert metrics.kg_precision == 0.4
    assert metrics.kg_faithfulness == 0.625
    assert metrics.kg_faithfulness != metrics.kg_precision
    assert metrics.hallucination_rate == 0.4


def test_evaluate_benchmark_questions_reuses_fixed_question_set(monkeypatch) -> None:
    evaluator = KGAFEEvaluator(kg=KnowledgeGraph(), enable_judge_panel=False)
    questions = [
        BenchmarkQuestion(
            question_id="q1",
            question="Question 1",
            gold_answer="Gold 1",
            question_type="single_hop",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question_id="q2",
            question="Question 2",
            gold_answer="Gold 2",
            question_type="comparison",
            difficulty="medium",
        ),
    ]

    class StubQA:
        def query(self, question: str):
            return {"final_answer": f"answer for {question}"}

    def fake_evaluate_answer(question, answer, gold_answer=None, relevant_triples=None):
        return EvaluationResult(
            question=question,
            answer=answer,
            gold_answer=gold_answer,
            metrics=KGAFEMetrics(kgafe_score=0.5, kg_precision=0.4, kg_faithfulness=0.6),
        )

    monkeypatch.setattr(evaluator, "evaluate_answer", fake_evaluate_answer)

    result = evaluator.evaluate_benchmark_questions(questions, qa_system=StubQA())

    assert len(result.individual_results) == 2
    assert result.individual_results[0].question_id == "q1"
    assert result.individual_results[0].question_type == "single_hop"
    assert result.individual_results[1].question_type == "comparison"


def test_generate_benchmark_questions_respects_requested_count() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("a", ["A"], "TYPE")
    kg.add_entity("b", ["B"], "TYPE")
    kg.add_entity("c", ["C"], "TYPE")
    kg.add_triple("a", "REL", "b", 1.0)
    kg.add_triple("b", "REL", "c", 1.0)
    evaluator = KGAFEEvaluator(kg=kg, enable_judge_panel=False)

    questions = evaluator.generate_benchmark_questions(
        n_questions=2,
        question_types=["single_hop", "multi_hop", "negative"],
    )

    assert len(questions) <= 2


def test_generate_benchmark_questions_filters_low_quality_entities() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("lowest_quartile", ["lowest quartile"], "CARDIOVASCULAR_MEASUREMENT")
    kg.add_entity("il6", ["IL-6"], "BIOMARKER")
    kg.add_entity("microvascular_dysfunction", ["microvascular dysfunction"], "CONDITION")
    kg.add_triple("lowest_quartile", "ASSOCIATED_WITH", "il6", 1.0)
    kg.add_triple("il6", "ASSOCIATED_WITH", "microvascular_dysfunction", 1.0)
    evaluator = KGAFEEvaluator(kg=kg, enable_judge_panel=False)

    questions = evaluator.generate_benchmark_questions(
        n_questions=3,
        question_types=["single_hop", "negative"],
    )

    serialized = [question.to_dict() for question in questions]
    assert all("lowest quartile" not in q["question"].lower() for q in serialized)


def test_tier2_path_inference_handles_null_best_path_index(monkeypatch) -> None:
    kg = KnowledgeGraph()
    kg.add_entity("a", ["A"], "TYPE")
    kg.add_entity("b", ["B"], "TYPE")
    kg.add_entity("c", ["C"], "TYPE")
    kg.add_triple("a", "REL_1", "b", 1.0)
    kg.add_triple("b", "REL_2", "c", 1.0)

    verifier = TripleVerifier(kg=kg)

    def fake_chat_completion_json(*args, **kwargs):
        return {
            "verdict": "supported",
            "confidence": 0.9,
            "reasoning": "The path supports the fact.",
            "best_path_index": None,
        }

    monkeypatch.setattr(
        "evaluation.kgafe.triple_verifier.chat_completion_json",
        fake_chat_completion_json,
    )

    result = verifier.verify(
        fact_id="f1",
        fact_text="A is related to C",
        entities_mentioned=["A", "C"],
        relation_implied="related_to",
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.tier == 2
    assert len(result.supporting_paths) == 1


def test_cross_domain_questions_are_generated_when_org_chart_is_available(monkeypatch) -> None:
    kg = KnowledgeGraph()
    kg.add_entity("method_a", ["Method A"], "METHOD")
    kg.add_entity("dataset_b", ["Dataset B"], "DATASET")
    kg.add_entity("metric_c", ["Metric C"], "METRIC")
    kg.add_triple("method_a", "USED_FOR", "dataset_b", 1.0)
    kg.add_triple("dataset_b", "EVALUATE_FOR", "metric_c", 1.0)

    methods = Domain(
        domain_id="methods",
        label="Methods",
        description="Method domain",
        entity_ids=["method_a"],
        relation_schema=["USED_FOR"],
        topics=[],
    )
    datasets = Domain(
        domain_id="datasets",
        label="Datasets",
        description="Dataset domain",
        entity_ids=["dataset_b"],
        relation_schema=["USED_FOR", "EVALUATE_FOR"],
        topics=[],
    )
    metrics = Domain(
        domain_id="metrics",
        label="Metrics",
        description="Metric domain",
        entity_ids=["metric_c"],
        relation_schema=["EVALUATE_FOR"],
        topics=[],
    )
    org_chart = OrgChart(
        domains=[methods, datasets, metrics],
        cross_domain_relations=kg.triples,
    )

    gen = BenchmarkGenerator(kg=kg, org_chart=org_chart)
    monkeypatch.setattr(gen, "_phrase_question", lambda template, subj, obj, rel: template)

    questions = gen.generate(n_questions=1, question_types=["cross_domain"])

    assert len(questions) == 1
    assert questions[0].question_type == "cross_domain"
    assert len(questions[0].expected_domains) >= 2
