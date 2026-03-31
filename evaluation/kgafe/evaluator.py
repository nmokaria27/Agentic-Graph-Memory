"""
KGAFE Evaluator — the main evaluation orchestrator.

Ties together:
  - AtomicDecomposer: breaks answers into atomic facts
  - TripleVerifier: three-tier KG verification
  - JudgePanel: multi-agent LLM evaluation
  - BenchmarkGenerator: auto-generated evaluation sets

Computes the following novel metrics:
  - KG-Faithfulness: fraction of atomic facts supported by KG (like FActScore, but KG-grounded)
  - KG-Precision: supported facts / total facts (no hallucination)
  - Hallucination Rate: contradicted + unverifiable facts / total
  - Path Validity: fraction verified via multi-hop paths (structural reasoning)
  - Coverage: fraction of relevant KG triples reflected in the answer
  - Groundedness Score: from judge panel
  - Tier Distribution: breakdown of which verification tier resolved each fact
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.domain_experts import neighbourhood

from .atomic_decomposer import AtomicDecomposer, AtomicFact
from .triple_verifier import TripleVerifier, VerificationResult, Verdict
from .judge_panel import JudgePanel, PanelVerdict
from .benchmark_generator import BenchmarkGenerator, BenchmarkQuestion


@dataclass
class KGAFEMetrics:
    """Complete KGAFE evaluation metrics for a single QA pair."""

    # Core metrics
    kg_faithfulness: float = 0.0  # supported / (supported + contradicted + partially)
    kg_precision: float = 0.0  # supported / total_facts
    hallucination_rate: float = 0.0  # (contradicted + unverifiable) / total_facts
    path_validity: float = 0.0  # facts verified via path (tier 2) / total
    coverage: float = 0.0  # relevant KG triples mentioned / total relevant
    groundedness_score: float = 0.0  # from judge panel

    # Tier distribution
    tier1_count: int = 0  # exact match
    tier2_count: int = 0  # path-based
    tier3_count: int = 0  # semantic

    # Verdict distribution
    supported_count: int = 0
    contradicted_count: int = 0
    partially_supported_count: int = 0
    unverifiable_count: int = 0
    total_facts: int = 0

    # Judge panel scores
    correctness_score: float = 0.0
    completeness_score: float = 0.0
    judge_overall: float = 0.0

    # Composite
    kgafe_score: float = 0.0  # Weighted composite of all metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kgafe_score": round(self.kgafe_score, 4),
            "kg_faithfulness": round(self.kg_faithfulness, 4),
            "kg_precision": round(self.kg_precision, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "path_validity": round(self.path_validity, 4),
            "coverage": round(self.coverage, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "correctness_score": round(self.correctness_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "judge_overall": round(self.judge_overall, 4),
            "tier_distribution": {
                "exact_match": self.tier1_count,
                "path_based": self.tier2_count,
                "semantic": self.tier3_count,
            },
            "verdict_distribution": {
                "supported": self.supported_count,
                "contradicted": self.contradicted_count,
                "partially_supported": self.partially_supported_count,
                "unverifiable": self.unverifiable_count,
                "total": self.total_facts,
            },
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single QA pair."""

    question: str
    answer: str
    metrics: KGAFEMetrics
    atomic_facts: List[Dict[str, Any]] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    judge_verdict: Optional[Dict[str, Any]] = None
    gold_answer: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "metrics": self.metrics.to_dict(),
            "atomic_facts": self.atomic_facts,
            "verification_results": self.verification_results,
            "judge_verdict": self.judge_verdict,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class BenchmarkResult:
    """Aggregate results across a full benchmark."""

    individual_results: List[EvaluationResult] = field(default_factory=list)
    aggregate_metrics: Optional[Dict[str, Any]] = None

    def compute_aggregates(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all evaluated questions."""
        if not self.individual_results:
            return {}

        n = len(self.individual_results)
        metrics_list = [r.metrics for r in self.individual_results]

        def avg(field_name: str) -> float:
            return sum(getattr(m, field_name) for m in metrics_list) / n

        self.aggregate_metrics = {
            "num_questions": n,
            "avg_kgafe_score": round(avg("kgafe_score"), 4),
            "avg_kg_faithfulness": round(avg("kg_faithfulness"), 4),
            "avg_kg_precision": round(avg("kg_precision"), 4),
            "avg_hallucination_rate": round(avg("hallucination_rate"), 4),
            "avg_path_validity": round(avg("path_validity"), 4),
            "avg_coverage": round(avg("coverage"), 4),
            "avg_groundedness": round(avg("groundedness_score"), 4),
            "avg_correctness": round(avg("correctness_score"), 4),
            "avg_completeness": round(avg("completeness_score"), 4),
            "total_facts_evaluated": sum(m.total_facts for m in metrics_list),
            "total_supported": sum(m.supported_count for m in metrics_list),
            "total_contradicted": sum(m.contradicted_count for m in metrics_list),
            "total_unverifiable": sum(m.unverifiable_count for m in metrics_list),
            "by_question_type": self._by_question_type(),
        }
        return self.aggregate_metrics

    def _by_question_type(self) -> Dict[str, Dict[str, float]]:
        """Break down metrics by question type (if gold answers have types)."""
        by_type: Dict[str, List[KGAFEMetrics]] = {}
        for r in self.individual_results:
            # Try to get question type from gold answer metadata
            qtype = "unknown"
            if r.gold_answer:
                # The gold_answer field stores the type when from benchmark
                for bq_type in ["single_hop", "multi_hop", "aggregation", "comparison", "negative"]:
                    if bq_type in str(getattr(r, "_question_type", "")):
                        qtype = bq_type
                        break
            by_type.setdefault(qtype, []).append(r.metrics)

        result = {}
        for qtype, metrics_list in by_type.items():
            n = len(metrics_list)
            result[qtype] = {
                "count": n,
                "avg_kgafe_score": round(
                    sum(m.kgafe_score for m in metrics_list) / n, 4
                ),
                "avg_faithfulness": round(
                    sum(m.kg_faithfulness for m in metrics_list) / n, 4
                ),
            }
        return result

    def to_dict(self) -> Dict[str, Any]:
        if not self.aggregate_metrics:
            self.compute_aggregates()
        return {
            "aggregate_metrics": self.aggregate_metrics,
            "individual_results": [r.to_dict() for r in self.individual_results],
        }


class KGAFEEvaluator:
    """
    Main KGAFE evaluation orchestrator.

    Usage:
        evaluator = KGAFEEvaluator(kg)

        # Evaluate a single QA pair
        result = evaluator.evaluate_answer(question, answer)

        # Run full benchmark
        benchmark = evaluator.run_benchmark(n_questions=50)
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "gemma3:27b",
        enable_judge_panel: bool = True,
    ):
        self.kg = kg
        self.model = model
        self.enable_judge_panel = enable_judge_panel

        # Initialize components
        self.decomposer = AtomicDecomposer(model=model)
        self.verifier = TripleVerifier(kg=kg, model=model)
        self.judge = JudgePanel(model=model) if enable_judge_panel else None
        self.benchmark_gen = BenchmarkGenerator(kg=kg, model=model)

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        gold_answer: Optional[str] = None,
        relevant_triples: Optional[List[Triple]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single QA answer using the full KGAFE pipeline.

        Args:
            question: The question that was asked
            answer: The generated answer to evaluate
            gold_answer: Optional gold-standard answer for comparison
            relevant_triples: Optional pre-identified relevant triples

        Returns:
            EvaluationResult with detailed metrics
        """
        start_time = time.time()

        entity_names = [
            entity.labels[0] if entity.labels else eid
            for eid, entity in self.kg.entities.items()
        ]

        # Step 1: Atomic fact decomposition
        print("  [KGAFE] Decomposing answer into atomic facts...")
        atomic_facts = self.decomposer.decompose(answer, entity_names)
        print(f"  [KGAFE] Extracted {len(atomic_facts)} atomic facts")

        # Step 2: Three-tier verification of each fact
        print("  [KGAFE] Verifying atomic facts against KG...")
        verification_results: List[VerificationResult] = []
        for fact in atomic_facts:
            vr = self.verifier.verify(
                fact_id=fact.fact_id,
                fact_text=fact.text,
                entities_mentioned=fact.entities_mentioned,
                relation_implied=fact.relation_implied,
            )
            verification_results.append(vr)
            status = f"{'✓' if vr.verdict == Verdict.SUPPORTED else '✗' if vr.verdict == Verdict.CONTRADICTED else '?'}"
            print(f"    {status} [{vr.verdict.value}] T{vr.tier} ({vr.confidence:.2f}): {fact.text[:80]}")

        # Step 3: Judge panel evaluation
        judge_verdict = None
        if self.judge:
            print("  [KGAFE] Running judge panel...")
            kg_evidence = self._get_relevant_evidence(question, answer)
            judge_verdict = self.judge.evaluate(
                question=question,
                answer=answer,
                kg_evidence=kg_evidence,
                atomic_facts=[f.to_dict() for f in atomic_facts],
                verification_results=[vr.to_dict() for vr in verification_results],
            )
            print(f"  [KGAFE] Judge panel: {judge_verdict.overall_score:.2f} "
                  f"({'PASS' if judge_verdict.overall_pass else 'FAIL'})")

        # Step 4: Compute metrics
        metrics = self._compute_metrics(
            atomic_facts, verification_results, judge_verdict,
            question, answer, relevant_triples,
        )

        duration = time.time() - start_time

        return EvaluationResult(
            question=question,
            answer=answer,
            metrics=metrics,
            atomic_facts=[f.to_dict() for f in atomic_facts],
            verification_results=[vr.to_dict() for vr in verification_results],
            judge_verdict=judge_verdict.to_dict() if judge_verdict else None,
            gold_answer=gold_answer,
            duration_seconds=duration,
        )

    def run_benchmark(
        self,
        n_questions: int = 50,
        qa_system=None,
        question_types: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """
        Generate a benchmark and evaluate the QA system against it.

        Args:
            n_questions: Number of benchmark questions to generate
            qa_system: The QAOrchestrator to evaluate (must have .query() method)
            question_types: Which question types to include

        Returns:
            BenchmarkResult with individual and aggregate metrics
        """
        print(f"\n{'='*70}")
        print(f"KGAFE BENCHMARK: Generating {n_questions} questions")
        print(f"{'='*70}\n")

        # Generate benchmark questions
        questions = self.benchmark_gen.generate(n_questions, question_types)
        print(f"Generated {len(questions)} benchmark questions")

        # Evaluate each question
        results = BenchmarkResult()
        for i, bq in enumerate(questions):
            print(f"\n--- Question {i+1}/{len(questions)} [{bq.question_type}] ---")
            print(f"Q: {bq.question}")

            if qa_system:
                # Get answer from QA system
                qa_result = qa_system.query(bq.question)
                answer = qa_result.get("final_answer", "")
            else:
                # No QA system — evaluate gold answer against itself (sanity check)
                answer = bq.gold_answer

            # Convert supporting triples to Triple objects
            relevant = [
                Triple(
                    subject=t["subject"],
                    relation=t["relation"],
                    object=t["object"],
                )
                for t in bq.supporting_triples
            ]

            # Evaluate
            eval_result = self.evaluate_answer(
                question=bq.question,
                answer=answer,
                gold_answer=bq.gold_answer,
                relevant_triples=relevant,
            )
            eval_result._question_type = bq.question_type
            results.individual_results.append(eval_result)

            print(f"  KGAFE Score: {eval_result.metrics.kgafe_score:.3f}")

        # Compute aggregates
        agg = results.compute_aggregates()
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"  Questions evaluated: {agg['num_questions']}")
        print(f"  Avg KGAFE Score: {agg['avg_kgafe_score']:.4f}")
        print(f"  Avg Faithfulness: {agg['avg_kg_faithfulness']:.4f}")
        print(f"  Avg Precision: {agg['avg_kg_precision']:.4f}")
        print(f"  Avg Hallucination Rate: {agg['avg_hallucination_rate']:.4f}")
        print(f"  Avg Coverage: {agg['avg_coverage']:.4f}")
        print(f"  Avg Groundedness: {agg['avg_groundedness']:.4f}")
        print(f"{'='*70}\n")

        return results

    def _compute_metrics(
        self,
        facts: List[AtomicFact],
        verifications: List[VerificationResult],
        judge_verdict: Optional[PanelVerdict],
        question: str,
        answer: str,
        relevant_triples: Optional[List[Triple]],
    ) -> KGAFEMetrics:
        """Compute all KGAFE metrics from verification and judge results."""
        metrics = KGAFEMetrics()
        metrics.total_facts = len(facts)

        if not verifications:
            return metrics

        # Verdict counts
        for vr in verifications:
            if vr.verdict == Verdict.SUPPORTED:
                metrics.supported_count += 1
            elif vr.verdict == Verdict.CONTRADICTED:
                metrics.contradicted_count += 1
            elif vr.verdict == Verdict.PARTIALLY_SUPPORTED:
                metrics.partially_supported_count += 1
            else:
                metrics.unverifiable_count += 1

            # Tier counts
            if vr.tier == 1:
                metrics.tier1_count += 1
            elif vr.tier == 2:
                metrics.tier2_count += 1
            else:
                metrics.tier3_count += 1

        total = metrics.total_facts
        if total == 0:
            return metrics

        # Core metrics
        metrics.kg_faithfulness = metrics.supported_count / total
        metrics.kg_precision = metrics.supported_count / total
        metrics.hallucination_rate = (
            metrics.contradicted_count + metrics.unverifiable_count
        ) / total
        metrics.path_validity = metrics.tier2_count / total

        # Coverage: what fraction of relevant KG triples are reflected in the answer
        if relevant_triples:
            mentioned = 0
            answer_lower = answer.lower()
            for t in relevant_triples:
                subj_name = self._entity_label(t.subject).lower()
                obj_name = self._entity_label(t.object).lower()
                if subj_name in answer_lower and obj_name in answer_lower:
                    mentioned += 1
            metrics.coverage = mentioned / max(len(relevant_triples), 1)
        else:
            # Estimate coverage from entity mentions
            metrics.coverage = self._estimate_coverage(question, answer)

        # Judge panel scores
        if judge_verdict:
            for v in judge_verdict.individual_verdicts:
                if v.dimension == "correctness":
                    metrics.correctness_score = v.score
                elif v.dimension == "completeness":
                    metrics.completeness_score = v.score
                elif v.dimension == "groundedness":
                    metrics.groundedness_score = v.score
            metrics.judge_overall = judge_verdict.overall_score

        # Composite KGAFE score
        # Weighted combination emphasizing faithfulness and groundedness
        metrics.kgafe_score = (
            0.30 * metrics.kg_faithfulness
            + 0.25 * (1.0 - metrics.hallucination_rate)
            + 0.20 * metrics.groundedness_score
            + 0.15 * metrics.coverage
            + 0.10 * metrics.correctness_score
        )

        return metrics

    def _get_relevant_evidence(self, question: str, answer: str) -> str:
        """Gather relevant KG evidence for the judge panel."""
        import re

        # Extract entity mentions from both question and answer
        text = f"{question} {answer}".lower()
        relevant_triples = []

        for eid, entity in self.kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                if len(n) < 3:
                    continue
                pattern = r'\b' + re.escape(n.lower()) + r'\b'
                if re.search(pattern, text):
                    # Get neighbourhood
                    nbr = neighbourhood(self.kg, eid, hops=2)
                    relevant_triples.extend(nbr)
                    break

        # Deduplicate
        seen = set()
        unique = []
        for t in relevant_triples:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                unique.append(t)

        lines = [
            f"({t.subject}) -[{t.relation}]-> ({t.object})"
            for t in unique[:80]
        ]
        return "\n".join(lines) if lines else "No relevant KG evidence found."

    def _estimate_coverage(self, question: str, answer: str) -> float:
        """Estimate answer coverage when no gold triples are available."""
        import re
        text = f"{question}".lower()

        # Find entities mentioned in the question
        query_entities = []
        for eid, entity in self.kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                if len(n) < 3:
                    continue
                if re.search(r'\b' + re.escape(n.lower()) + r'\b', text):
                    query_entities.append(eid)
                    break

        if not query_entities:
            return 0.5  # Can't estimate

        # Count how many triples involving query entities are reflected in the answer
        answer_lower = answer.lower()
        relevant = 0
        mentioned = 0
        for t in self.kg.triples:
            if t.subject in query_entities or t.object in query_entities:
                relevant += 1
                subj_label = self._entity_label(t.subject).lower()
                obj_label = self._entity_label(t.object).lower()
                if subj_label in answer_lower or obj_label in answer_lower:
                    mentioned += 1

        return mentioned / max(relevant, 1)

    def _entity_label(self, entity_id: str) -> str:
        """Get best label for an entity."""
        entity = self.kg.entities.get(entity_id)
        if entity and entity.labels:
            return entity.labels[0]
        return entity_id.replace("_", " ")
