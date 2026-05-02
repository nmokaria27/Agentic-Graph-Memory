"""Answer-support coverage diagnostics for QA-oriented KG construction."""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from multi_agent_kg.core.governance import OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def content_words(text: str) -> List[str]:
    return [
        token
        for token in normalize_text(text).split()
        if len(token) >= 3
    ]


def contains_normalized(haystack: str, needle: str) -> bool:
    needle_norm = normalize_text(needle)
    if not needle_norm:
        return False
    return needle_norm in normalize_text(haystack)


def triple_text(triple: Triple) -> str:
    metadata = triple.metadata if isinstance(triple.metadata, dict) else {}
    parts = [triple.subject, triple.relation, triple.object, triple.source or ""]
    for key in ("evidence", "evidence_text", "evidence_span", "source_text", "source_sentence", "snippet"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


@dataclass
class SupportCoverageResult:
    question_id: str
    question: str
    gold_answer: str
    answer_in_entity: bool
    answer_in_triple: bool
    answer_in_evidence: bool
    answer_in_domain_memory: bool
    question_entity_coverage: float
    matched_entities: List[str]
    matched_triples: List[str]
    matched_domains: List[str]
    likely_failure_stage: str

    @property
    def answer_supported(self) -> bool:
        return (
            self.answer_in_entity
            or self.answer_in_triple
            or self.answer_in_evidence
            or self.answer_in_domain_memory
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["answer_supported"] = self.answer_supported
        return payload


def evaluate_question_support(
    *,
    kg: KnowledgeGraph,
    question_id: str,
    question: str,
    gold_answer: str,
    org_chart: Optional[OrgChart] = None,
) -> SupportCoverageResult:
    answer_in_entity = False
    answer_in_triple = False
    answer_in_evidence = False
    answer_in_domain_memory = False
    boolean_answer_supported = False
    matched_entities: List[str] = []
    matched_triples: List[str] = []
    matched_domains: List[str] = []

    question_terms = set(content_words(question))
    question_matched_entity_ids: List[str] = []
    for entity_id, entity in kg.entities.items():
        labels = [entity_id, *(entity.labels or [])]
        if any(contains_normalized(label, gold_answer) for label in labels):
            answer_in_entity = True
            matched_entities.append(entity_id)
        label_terms = set(content_words(" ".join(labels)))
        if question_terms and question_terms & label_terms:
            question_matched_entity_ids.append(entity_id)

    for triple in kg.triples:
        edge_text = f"{triple.subject} {triple.relation} {triple.object}"
        if contains_normalized(edge_text, gold_answer):
            answer_in_triple = True
            matched_triples.append(f"({triple.subject}) -[{triple.relation}]-> ({triple.object})")
        if contains_normalized(triple_text(triple), gold_answer):
            answer_in_evidence = True
            fact = f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"
            if fact not in matched_triples:
                matched_triples.append(fact)

    if org_chart is not None:
        for domain in org_chart.domains:
            summary = domain.memory_card_summary()
            if contains_normalized(summary, gold_answer):
                answer_in_domain_memory = True
                matched_domains.append(domain.domain_id)

    if question_terms:
        entity_texts = {
            token
            for entity_id, entity in kg.entities.items()
            for token in content_words(" ".join([entity_id, *(entity.labels or [])]))
        }
        question_entity_coverage = len(question_terms & entity_texts) / len(question_terms)
    else:
        question_entity_coverage = 0.0

    if normalize_text(gold_answer) in {"yes", "no"}:
        matched_set = set(question_matched_entity_ids)
        incident_triples = [
            triple for triple in kg.triples
            if triple.subject in matched_set or triple.object in matched_set
        ]
        connected_pairs = 0
        for index, entity_id in enumerate(question_matched_entity_ids):
            for other_id in question_matched_entity_ids[index + 1:]:
                if any(
                    {triple.subject, triple.object} == {entity_id, other_id}
                    or (
                        (triple.subject == entity_id or triple.object == entity_id)
                        and any(
                            second.subject == other_id
                            or second.object == other_id
                            or second.subject == triple.subject
                            or second.object == triple.object
                            for second in incident_triples
                        )
                    )
                    for triple in incident_triples
                ):
                    connected_pairs += 1
        boolean_answer_supported = (
            len(question_matched_entity_ids) >= 2
            and bool(incident_triples)
            and (connected_pairs > 0 or question_entity_coverage >= 0.5)
        )

    answer_supported = (
        answer_in_entity
        or answer_in_triple
        or answer_in_evidence
        or answer_in_domain_memory
        or boolean_answer_supported
    )
    if not answer_supported:
        likely_failure_stage = "extraction_miss"
    elif org_chart is not None and not matched_domains:
        likely_failure_stage = "domain_memory_or_routing_gap"
    else:
        likely_failure_stage = "retrieval_or_synthesis_gap"

    return SupportCoverageResult(
        question_id=question_id,
        question=question,
        gold_answer=gold_answer,
        answer_in_entity=answer_in_entity,
        answer_in_triple=answer_in_triple,
        answer_in_evidence=answer_in_evidence,
        answer_in_domain_memory=answer_in_domain_memory,
        question_entity_coverage=round(question_entity_coverage, 4),
        matched_entities=matched_entities[:10],
        matched_triples=matched_triples[:10],
        matched_domains=matched_domains[:10],
        likely_failure_stage=likely_failure_stage,
    )


def aggregate_support_results(results: Iterable[SupportCoverageResult]) -> Dict[str, Any]:
    rows = list(results)
    if not rows:
        return {
            "num_questions": 0,
            "answer_support_rate": 0.0,
            "answer_entity_rate": 0.0,
            "answer_triple_rate": 0.0,
            "answer_evidence_rate": 0.0,
            "answer_domain_memory_rate": 0.0,
            "avg_question_entity_coverage": 0.0,
            "failure_stage_counts": {},
        }
    failure_counts: Dict[str, int] = {}
    for row in rows:
        failure_counts[row.likely_failure_stage] = failure_counts.get(row.likely_failure_stage, 0) + 1
    total = len(rows)
    return {
        "num_questions": total,
        "answer_support_rate": round(sum(row.answer_supported for row in rows) / total, 4),
        "answer_entity_rate": round(sum(row.answer_in_entity for row in rows) / total, 4),
        "answer_triple_rate": round(sum(row.answer_in_triple for row in rows) / total, 4),
        "answer_evidence_rate": round(sum(row.answer_in_evidence for row in rows) / total, 4),
        "answer_domain_memory_rate": round(sum(row.answer_in_domain_memory for row in rows) / total, 4),
        "avg_question_entity_coverage": round(
            sum(row.question_entity_coverage for row in rows) / total,
            4,
        ),
        "failure_stage_counts": failure_counts,
    }
