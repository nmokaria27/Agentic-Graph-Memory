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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.domain_experts import OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.domain_experts import neighbourhood
from multi_agent_kg.core.kg_operations import normalize_entity_name

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
    question_id: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    atomic_facts: List[Dict[str, Any]] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    judge_verdict: Optional[Dict[str, Any]] = None
    gold_answer: Optional[str] = None
    aux_metrics: Dict[str, Any] = field(default_factory=dict)
    system_metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "metrics": self.metrics.to_dict(),
            "atomic_facts": self.atomic_facts,
            "verification_results": self.verification_results,
            "judge_verdict": self.judge_verdict,
            "aux_metrics": self.aux_metrics,
            "system_metadata": self.system_metadata,
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
            "aux_metrics": self._aggregate_aux_metrics(),
            "by_question_type": self._by_question_type(),
        }
        return self.aggregate_metrics

    def _aggregate_aux_metrics(self) -> Dict[str, float]:
        numeric_values: Dict[str, List[float]] = {}
        for result in self.individual_results:
            for key, value in result.aux_metrics.items():
                if isinstance(value, (int, float)):
                    numeric_values.setdefault(key, []).append(float(value))

        return {
            key: round(sum(values) / len(values), 4)
            for key, values in numeric_values.items()
            if values
        }

    def _by_question_type(self) -> Dict[str, Dict[str, float]]:
        """Break down metrics by question type (if gold answers have types)."""
        by_type: Dict[str, List[KGAFEMetrics]] = {}
        for r in self.individual_results:
            qtype = r.question_type or "unknown"
            by_type.setdefault(qtype, []).append(r.metrics)

        result = {}
        for qtype, metrics_list in by_type.items():
            q_results = [
                r for r in self.individual_results
                if (r.question_type or "unknown") == qtype
            ]
            n = len(metrics_list)
            aux_numeric: Dict[str, List[float]] = {}
            for r in q_results:
                for key, value in r.aux_metrics.items():
                    if isinstance(value, (int, float)):
                        aux_numeric.setdefault(key, []).append(float(value))
            result[qtype] = {
                "count": n,
                "avg_kgafe_score": round(
                    sum(m.kgafe_score for m in metrics_list) / n, 4
                ),
                "avg_faithfulness": round(
                    sum(m.kg_faithfulness for m in metrics_list) / n, 4
                ),
                "aux_metrics": {
                    key: round(sum(values) / len(values), 4)
                    for key, values in aux_numeric.items()
                    if values
                },
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
        org_chart: Optional[OrgChart] = None,
    ):
        self.kg = kg
        self.model = model
        self.enable_judge_panel = enable_judge_panel

        # Initialize components
        self.decomposer = AtomicDecomposer(model=model)
        self.verifier = TripleVerifier(kg=kg, model=model)
        self.judge = JudgePanel(model=model) if enable_judge_panel else None
        self.benchmark_gen = BenchmarkGenerator(kg=kg, model=model, org_chart=org_chart)

    def generate_benchmark_questions(
        self,
        n_questions: int = 50,
        question_types: Optional[List[str]] = None,
    ) -> List[BenchmarkQuestion]:
        """Generate a fixed benchmark set so multiple systems can be compared fairly."""
        return self.benchmark_gen.generate(n_questions, question_types)

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
        normalized_answer = self._normalize_answer_for_evaluation(answer)

        # Step 1: Atomic fact decomposition
        print("  [KGAFE] Decomposing answer into atomic facts...")
        atomic_facts = self.decomposer.decompose(normalized_answer, entity_names)
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
            kg_evidence = self._get_relevant_evidence(question, normalized_answer)
            judge_verdict = self.judge.evaluate(
                question=question,
                answer=normalized_answer,
                kg_evidence=kg_evidence,
                atomic_facts=[f.to_dict() for f in atomic_facts],
                verification_results=[vr.to_dict() for vr in verification_results],
            )
            print(f"  [KGAFE] Judge panel: {judge_verdict.overall_score:.2f} "
                  f"({'PASS' if judge_verdict.overall_pass else 'FAIL'})")

        # Step 4: Compute metrics
        metrics = self._compute_metrics(
            atomic_facts, verification_results, judge_verdict,
            question, normalized_answer, relevant_triples,
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

    def evaluate_benchmark_questions(
        self,
        questions: List[BenchmarkQuestion],
        qa_system=None,
    ) -> BenchmarkResult:
        """Evaluate one QA system on a fixed benchmark question set."""
        results = BenchmarkResult()
        for i, bq in enumerate(questions):
            print(f"\n--- Question {i+1}/{len(questions)} [{bq.question_type}] ---")
            print(f"Q: {bq.question}")

            qa_result: Dict[str, Any] = {}
            if qa_system:
                # Get answer from QA system
                if hasattr(qa_system, "query_benchmark_question"):
                    qa_result = qa_system.query_benchmark_question(bq)
                else:
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
            eval_result.question_id = bq.question_id
            eval_result.question_type = bq.question_type
            eval_result.difficulty = bq.difficulty
            eval_result.system_metadata = self._extract_system_metadata(qa_result)
            eval_result.aux_metrics = self._compute_aux_metrics(
                question=bq,
                answer=answer,
                system_metadata=eval_result.system_metadata,
            )
            self._apply_negative_abstention_credit(
                eval_result=eval_result,
                benchmark_question=bq,
            )
            results.individual_results.append(eval_result)

            print(f"  KGAFE Score: {eval_result.metrics.kgafe_score:.3f}")

        return results

    def _apply_negative_abstention_credit(
        self,
        eval_result: EvaluationResult,
        benchmark_question: BenchmarkQuestion,
    ) -> None:
        """
        Credit correct abstention on negative questions.

        Negative benchmark questions are generated so that the correct answer is
        effectively "no such relationship exists in the KG". A cautious answer
        like "there is no evidence" or "cannot determine from the KG" should not
        be penalized as a hallucination simply because the atomic decomposer emits
        zero facts or an unverifiable negation fact.
        """
        if benchmark_question.question_type != "negative":
            return
        if eval_result.aux_metrics.get("negative_abstention") != 1.0:
            return

        metrics = eval_result.metrics
        total = max(metrics.total_facts, 1)

        metrics.total_facts = total
        metrics.supported_count = total
        metrics.partially_supported_count = 0
        metrics.contradicted_count = 0
        metrics.unverifiable_count = 0

        metrics.kg_faithfulness = 1.0
        metrics.kg_precision = 1.0
        metrics.hallucination_rate = 0.0
        metrics.path_validity = 0.0
        metrics.coverage = max(metrics.coverage, 1.0)

        if self.judge:
            if metrics.correctness_score == 0.0:
                metrics.correctness_score = 1.0
            if metrics.completeness_score == 0.0:
                metrics.completeness_score = 1.0
            if metrics.groundedness_score == 0.0:
                metrics.groundedness_score = 1.0
            if metrics.judge_overall == 0.0:
                metrics.judge_overall = (
                    metrics.correctness_score
                    + metrics.completeness_score
                    + metrics.groundedness_score
                ) / 3.0

        metrics.kgafe_score = (
            0.30 * metrics.kg_faithfulness
            + 0.25 * (1.0 - metrics.hallucination_rate)
            + 0.20 * metrics.groundedness_score
            + 0.15 * metrics.coverage
            + 0.10 * metrics.correctness_score
        )

    def _extract_system_metadata(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract routing/provenance metadata from a QA system result when available."""
        if not qa_result:
            return {}

        domain_responses = qa_result.get("domain_responses") or []
        consulted_domains = []
        for response in domain_responses:
            domain_id = response.get("domain_id")
            if domain_id and domain_id not in consulted_domains:
                consulted_domains.append(domain_id)

        exploration = qa_result.get("exploration_summary") or {}
        for domain_id in exploration.get("domains_explored", []):
            if domain_id and domain_id not in consulted_domains:
                consulted_domains.append(domain_id)

        return {
            "consulted_domains": consulted_domains,
            "num_consulted_domains": len(consulted_domains),
            "overall_confidence": qa_result.get("overall_confidence"),
            "overall_coverage": qa_result.get("overall_coverage"),
        }

    def _compute_aux_metrics(
        self,
        question: BenchmarkQuestion,
        answer: str,
        system_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute auxiliary hard metrics from benchmark metadata."""
        answer_norm = normalize_entity_name(answer)
        aux: Dict[str, Any] = {
            "answer_word_count": len((answer or "").split()),
        }

        entity_labels = [
            normalize_entity_name(self._entity_label(entity_id))
            for entity_id in question.entities_involved
        ]
        entity_labels = [label for label in entity_labels if label]
        if entity_labels:
            mentioned = sum(1 for label in entity_labels if label in answer_norm)
            aux["entity_recall"] = mentioned / len(entity_labels)

        if question.supporting_triples:
            covered = 0
            for triple in question.supporting_triples:
                subj = normalize_entity_name(self._entity_label(triple["subject"]))
                obj = normalize_entity_name(self._entity_label(triple["object"]))
                if subj in answer_norm and obj in answer_norm:
                    covered += 1
            aux["support_triple_recall"] = covered / len(question.supporting_triples)

        if question.supporting_paths:
            path_entities = []
            for path in question.supporting_paths:
                for triple in path:
                    path_entities.append(triple["subject"])
                    path_entities.append(triple["object"])
            unique_entities = list(dict.fromkeys(path_entities))
            if unique_entities:
                mentioned = 0
                for entity_id in unique_entities:
                    label = normalize_entity_name(self._entity_label(entity_id))
                    if label and label in answer_norm:
                        mentioned += 1
                aux["path_entity_recall"] = mentioned / len(unique_entities)

        expected_domains = question.expected_domains or []
        consulted_domains = system_metadata.get("consulted_domains") or []
        if expected_domains and consulted_domains:
            overlap = len(set(expected_domains) & set(consulted_domains))
            aux["expected_domain_recall"] = overlap / len(expected_domains)
            aux["expected_domain_precision"] = overlap / len(consulted_domains)

        if question.question_type == "negative":
            abstain_cues = (
                " no ", " not ", "does not", "do not", "cannot", "can't",
                "not contain", "no evidence", "not directly support", "unknown",
            )
            padded = f" {answer.lower()} "
            aux["negative_abstention"] = 1.0 if any(cue in padded for cue in abstain_cues) else 0.0

        return aux

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

        questions = self.generate_benchmark_questions(n_questions, question_types)
        print(f"Generated {len(questions)} benchmark questions")

        results = self.evaluate_benchmark_questions(questions, qa_system=qa_system)

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

        evaluable_total = (
            metrics.supported_count
            + metrics.partially_supported_count
            + metrics.contradicted_count
        )

        # Core metrics
        # Faithfulness gives partial credit for partially supported claims and
        # excludes fully unverifiable claims from the denominator.
        metrics.kg_faithfulness = (
            (metrics.supported_count + 0.5 * metrics.partially_supported_count)
            / evaluable_total
            if evaluable_total > 0
            else 0.0
        )
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
        text_norm = normalize_entity_name(f"{question} {answer}")
        relevant_triples = []

        for eid, entity in self.kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                n_norm = normalize_entity_name(n)
                if len(n_norm) < 3:
                    continue
                if n_norm in text_norm:
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
        text_norm = normalize_entity_name(question)

        # Find entities mentioned in the question
        query_entities = []
        for eid, entity in self.kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                n_norm = normalize_entity_name(n)
                if len(n_norm) < 3:
                    continue
                if n_norm in text_norm:
                    query_entities.append(eid)
                    break

        if not query_entities:
            return 0.5  # Can't estimate

        # Count how many triples involving query entities are reflected in the answer
        answer_norm = normalize_entity_name(answer)
        relevant = 0
        mentioned = 0
        for t in self.kg.triples:
            if t.subject in query_entities or t.object in query_entities:
                relevant += 1
                subj_norm = normalize_entity_name(self._entity_label(t.subject))
                obj_norm = normalize_entity_name(self._entity_label(t.object))
                if subj_norm in answer_norm or obj_norm in answer_norm:
                    mentioned += 1

        return mentioned / max(relevant, 1)

    def _entity_label(self, entity_id: str) -> str:
        """Get best label for an entity."""
        entity = self.kg.entities.get(entity_id)
        if entity and entity.labels:
            return entity.labels[0]
        return entity_id.replace("_", " ")

    def _normalize_answer_for_evaluation(self, answer: str) -> str:
        """Strip obvious agent-scaffolding before atomic decomposition."""
        cleaned = answer or ""
        cleaned = re.sub(r"\((?:global_expert|[a-z_]+_expert)\)", "", cleaned)
        cleaned = re.sub(r"\[(?:global_expert|[a-z_]+)\]", "", cleaned)

        boilerplate_prefixes = [
            r"^Based on the (?:available|provided )?knowledge graph,\s*",
            r"^According to the (?:available|provided )?knowledge graph,\s*",
            r"^The (?:available )?knowledge graph (?:shows|indicates|suggests) that\s*",
            r"^The graph (?:shows|indicates|suggests) that\s*",
        ]
        for pattern in boilerplate_prefixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return re.sub(r"\s+", " ", cleaned).strip()
