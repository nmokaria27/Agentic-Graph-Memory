"""
Governed knowledge graph wrapper.

Definition 1.
  A Governed Knowledge Graph is a tuple G = (E, T, D, phi, gamma) where E is
  the entity set, T is the triple set, D is the set of governed domains,
  phi:E -> 2^D maps entities to their owning domains, and gamma routes a
  proposed triple update to the responsible governing domain(s) before a
  decision in {approve, reject, revise, escalate, auto_approve} is committed.

This wrapper operationalizes that definition by composing a base
KnowledgeGraph with an OrgChart, an audit trail, and explicit propose/commit
governance semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional

from multi_agent_kg.core.governance import GovernanceAssignment, OrgChart, coerce_metadata
from multi_agent_kg.core.knowledge_graph import Entity, KnowledgeGraph, Triple


@dataclass
class GovernanceDecision:
    """
    A single governance decision over a proposed triple update.

    `committed` intentionally follows two-phase commit semantics:
    a decision is created first as an audit/provenance record, then
    `commit_decision()` mutates `committed=True` only if the triple is
    actually inserted into the governed graph.
    """

    triple: Triple
    action: str
    domain_id: Optional[str]
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    revised_triple: Optional[Triple] = None
    assignment: Optional[GovernanceAssignment] = None
    committed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple": {
                "subject": self.triple.subject,
                "relation": self.triple.relation,
                "object": self.triple.object,
                "confidence": self.triple.confidence,
                "source": self.triple.source,
                "metadata": self.triple.metadata,
            },
            "action": self.action,
            "domain_id": self.domain_id,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "revised_triple": (
                {
                    "subject": self.revised_triple.subject,
                    "relation": self.revised_triple.relation,
                    "object": self.revised_triple.object,
                    "confidence": self.revised_triple.confidence,
                    "source": self.revised_triple.source,
                    "metadata": self.revised_triple.metadata,
                }
                if self.revised_triple
                else None
            ),
            "assignment": self.assignment.to_dict() if self.assignment else None,
            "committed": self.committed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceDecision":
        triple_data = data["triple"]
        triple = Triple(
            subject=triple_data["subject"],
            relation=triple_data["relation"],
            object=triple_data["object"],
            confidence=triple_data.get("confidence"),
            source=triple_data.get("source"),
            metadata=triple_data.get("metadata", {}),
        )
        revised_data = data.get("revised_triple")
        revised = None
        if revised_data:
            revised = Triple(
                subject=revised_data["subject"],
                relation=revised_data["relation"],
                object=revised_data["object"],
                confidence=revised_data.get("confidence"),
                source=revised_data.get("source"),
                metadata=revised_data.get("metadata", {}),
            )
        assignment = None
        if data.get("assignment"):
            assignment = GovernanceAssignment(
                assignment_type=data["assignment"]["assignment_type"],
                primary_domain_id=data["assignment"].get("primary_domain_id"),
                domain_ids=data["assignment"].get("domain_ids", []),
                rationale=data["assignment"].get("rationale", ""),
                score_breakdown=data["assignment"].get("score_breakdown", {}),
            )
        return cls(
            triple=triple,
            action=data["action"],
            domain_id=data.get("domain_id"),
            rationale=data.get("rationale", ""),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            revised_triple=revised,
            assignment=assignment,
            committed=data.get("committed", False),
        )


class GovernedKnowledgeGraph:
    """Composition wrapper that adds governance and auditability to a KG."""

    def __init__(
        self,
        kg: Optional[KnowledgeGraph] = None,
        org_chart: Optional[OrgChart] = None,
        governance_mode: str = "strict",
        review_callback: Optional[
            Callable[[Triple, GovernanceAssignment, KnowledgeGraph, OrgChart], GovernanceDecision]
        ] = None,
        min_admission_confidence: Optional[float] = None,
        confidence_policy_label: str = "confidence_policy",
    ):
        self._kg = kg or KnowledgeGraph()
        self._org_chart = org_chart or OrgChart()
        self._governance_mode = governance_mode
        self._audit_log: List[GovernanceDecision] = []
        self._pending_review: List[Triple] = []
        self._review_callback = review_callback
        self._min_admission_confidence = min_admission_confidence
        self._confidence_policy_label = confidence_policy_label
        self._bootstrap_assignment_stats: Dict[str, Any] = {}
        self._triage_threshold = 0.7
        self._triage_stats: Dict[str, Any] = {
            "auto_approved_low_risk": 0,
            "reviewed": 0,
            "policy_rejected": 0,
            "escalated_without_callback": 0,
            "review_reasons": {},
        }

    @property
    def kg(self) -> KnowledgeGraph:
        return self._kg

    @property
    def org_chart(self) -> OrgChart:
        return self._org_chart

    @property
    def entities(self) -> Dict[str, Entity]:
        return self._kg.entities

    @property
    def triples(self) -> List[Triple]:
        return self._kg.triples

    @property
    def governance_mode(self) -> str:
        return self._governance_mode

    @property
    def audit_log(self) -> List[GovernanceDecision]:
        return self._audit_log

    def set_review_callback(
        self,
        callback: Optional[
            Callable[[Triple, GovernanceAssignment, KnowledgeGraph, OrgChart], GovernanceDecision]
        ],
    ) -> None:
        self._review_callback = callback

    def set_org_chart(self, org_chart: OrgChart) -> None:
        self._org_chart = org_chart
        self._org_chart.refresh_cross_domain_relations(self._kg)

    def bootstrap_domains(self, builder: Any) -> OrgChart:
        org_chart = builder.build(self._kg)
        self.set_org_chart(org_chart)
        return org_chart

    def assign_entity_to_domains(self, entity_id: str, domain_ids: List[str]) -> None:
        if not self._org_chart.domains:
            return
        self._org_chart.assign_entity(entity_id, domain_ids)
        self._org_chart.refresh_cross_domain_relations(self._kg)

    def set_bootstrap_assignment_stats(self, stats: Dict[str, Any]) -> None:
        self._bootstrap_assignment_stats = dict(stats)

    def add_entity(
        self,
        entity_id: str,
        labels: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        return self._kg.add_entity(
            entity_id=entity_id,
            labels=labels,
            entity_type=entity_type,
            metadata=metadata,
        )

    def propose_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GovernanceDecision:
        triple = Triple(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )
        assignment = self._org_chart.route_triple_for_governance(triple)

        policy_decision = self._admission_policy_decision(triple, assignment)
        if policy_decision is not None:
            self.commit_decision(policy_decision)
            return policy_decision

        if self._governance_mode == "audit_only":
            decision = GovernanceDecision(
                triple=triple,
                action="auto_approve",
                domain_id=assignment.primary_domain_id,
                rationale="Audit-only mode automatically records and accepts proposals.",
                assignment=assignment,
            )
            self.commit_decision(decision)
            return decision

        if self._governance_mode == "permissive":
            action = "approve" if assignment.assignment_type != "unowned" else "auto_approve"
            rationale = (
                "Permissive mode accepted the proposal after ownership routing."
                if assignment.assignment_type != "unowned"
                else "Permissive mode accepted an unowned proposal and logged it for audit."
            )
            decision = GovernanceDecision(
                triple=triple,
                action=action,
                domain_id=assignment.primary_domain_id,
                rationale=rationale,
                assignment=assignment,
            )
            self.commit_decision(decision)
            return decision

        if self._governance_mode == "triage":
            review_reason = self._triage_reason(triple, assignment)
            if review_reason is None:
                self._triage_stats["auto_approved_low_risk"] += 1
                decision = GovernanceDecision(
                    triple=triple,
                    action="auto_approve",
                    domain_id=assignment.primary_domain_id,
                    rationale=(
                        "Triaged governance auto-approved a low-risk proposal "
                        "after ownership routing."
                    ),
                    assignment=assignment,
                )
                self.commit_decision(decision)
                return decision

            if self._review_callback is not None:
                self._triage_stats["reviewed"] += 1
                self._triage_stats["review_reasons"][review_reason] = (
                    self._triage_stats["review_reasons"].get(review_reason, 0) + 1
                )
                decision = self._review_callback(triple, assignment, self._kg, self._org_chart)
                if decision.assignment is None:
                    decision.assignment = assignment
                if review_reason:
                    suffix = f" [triage_reason={review_reason}]"
                    decision.rationale = (decision.rationale or "").strip() + suffix
                self.commit_decision(decision)
                return decision

            self._triage_stats["escalated_without_callback"] += 1
            self._triage_stats["review_reasons"][review_reason] = (
                self._triage_stats["review_reasons"].get(review_reason, 0) + 1
            )
            decision = GovernanceDecision(
                triple=triple,
                action="escalate",
                domain_id=assignment.primary_domain_id,
                rationale=(
                    "Triaged governance flagged the proposal for explicit review "
                    f"({review_reason}), but no review callback was configured."
                ),
                assignment=assignment,
            )
            self._audit_log.append(decision)
            self._pending_review.append(triple)
            return decision

        if self._review_callback is not None:
            decision = self._review_callback(triple, assignment, self._kg, self._org_chart)
            if decision.assignment is None:
                decision.assignment = assignment
            self.commit_decision(decision)
            return decision

        if assignment.assignment_type == "unowned":
            decision = GovernanceDecision(
                triple=triple,
                action="escalate",
                domain_id=None,
                rationale="Strict mode requires explicit review for unowned triples.",
                assignment=assignment,
            )
            self._audit_log.append(decision)
            self._pending_review.append(triple)
            return decision

        decision = GovernanceDecision(
            triple=triple,
            action="escalate",
            domain_id=assignment.primary_domain_id,
            rationale="Strict mode queued the proposal for explicit domain review.",
            assignment=assignment,
        )
        self._audit_log.append(decision)
        self._pending_review.append(triple)
        return decision

    def commit_decision(self, decision: GovernanceDecision) -> Optional[Triple]:
        if decision not in self._audit_log:
            self._audit_log.append(decision)

        approved_actions = {"approve", "auto_approve", "revise"}
        if decision.action not in approved_actions:
            return None

        triple = decision.revised_triple or decision.triple
        result = self._kg.add_triple(
            subject=triple.subject,
            relation=triple.relation,
            obj=triple.object,
            confidence=triple.confidence,
            source=triple.source,
            metadata=triple.metadata,
        )
        decision.committed = result is not None
        if result is not None:
            self._org_chart.update_cross_domain_relation(result)
        return result

    def _triage_reason(
        self,
        triple: Triple,
        assignment: GovernanceAssignment,
    ) -> Optional[str]:
        """Return a review reason for risky triples, or ``None`` if low-risk."""
        confidence = triple.confidence if triple.confidence is not None else 0.0
        if confidence < self._triage_threshold:
            return "low_confidence"
        if assignment.assignment_type in {"cross_domain", "unowned"}:
            return assignment.assignment_type
        if triple.metadata.get("schema_novel"):
            return "schema_novel"
        if self._relation_outside_schema(triple.relation):
            return "schema_novel"
        if self._kg.find_conflicts([triple]):
            return "conflict"
        return None

    def _admission_policy_decision(
        self,
        triple: Triple,
        assignment: GovernanceAssignment,
    ) -> Optional[GovernanceDecision]:
        """Apply deterministic governance policies before domain review.

        This is intentionally part of admission governance, not evaluation.
        Fixed-schema benchmarks can configure a conservative confidence floor
        so weak triples are auditable rejections instead of silently filtered
        during scoring.
        """
        if self._min_admission_confidence is None:
            return None
        confidence = triple.confidence if triple.confidence is not None else 0.0
        if confidence >= self._min_admission_confidence:
            return None

        if self._governance_mode == "triage":
            self._triage_stats["policy_rejected"] = (
                self._triage_stats.get("policy_rejected", 0) + 1
            )
            reason_counts = self._triage_stats.setdefault("review_reasons", {})
            reason_counts[self._confidence_policy_label] = (
                reason_counts.get(self._confidence_policy_label, 0) + 1
            )

        return GovernanceDecision(
            triple=triple,
            action="reject",
            domain_id=assignment.primary_domain_id,
            rationale=(
                f"Governance admission policy rejected proposal: confidence "
                f"{confidence:.2f} is below required threshold "
                f"{self._min_admission_confidence:.2f} "
                f"[policy={self._confidence_policy_label}]"
            ),
            assignment=assignment,
        )

    def _relation_outside_schema(self, relation: str) -> bool:
        if not self._org_chart.domains:
            return False
        allowed_relations = set()
        for domain in self._org_chart.domains:
            allowed_relations.update(domain.relation_schema.keys())
            domain_metadata = coerce_metadata(domain.metadata)
            allowed_relations.update(domain_metadata.get("seed_relation_types", []))
        if not allowed_relations:
            return False
        if relation in allowed_relations:
            return False
        normalized = relation.upper().replace("-", "_").replace(" ", "_")
        for allowed in allowed_relations:
            if allowed.upper().replace("-", "_").replace(" ", "_") == normalized:
                return False
        return True

    def add_triple_bypass(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Triple]:
        triple = Triple(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )
        decision = GovernanceDecision(
            triple=triple,
            action="auto_approve",
            domain_id=None,
            rationale="Triple added through bypass path for backward compatibility.",
            assignment=None,
        )
        self._audit_log.append(decision)
        result = self._kg.add_triple(
            subject=subject,
            relation=relation,
            obj=obj,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
        decision.committed = result is not None
        if result is not None:
            self._org_chart.update_cross_domain_relation(result)
        return result

    def get_domain_subgraph(self, domain_id: str) -> Dict[str, Any]:
        domain = self._org_chart.find_domain(domain_id)
        if domain is None:
            return {"entities": [], "triples": []}
        entities, triples = domain.get_subgraph(self._kg)
        return {"entities": entities, "triples": triples}

    def get_governance_history(self, domain_id: Optional[str] = None) -> List[GovernanceDecision]:
        if domain_id is None:
            return list(self._audit_log)
        return [
            decision
            for decision in self._audit_log
            if decision.domain_id == domain_id
            or (
                decision.assignment is not None
                and domain_id in decision.assignment.domain_ids
            )
        ]

    def get_stats(self) -> Dict[str, Any]:
        action_counts: Dict[str, int] = {}
        assignment_counts: Dict[str, int] = {}
        domain_decisions: Dict[str, int] = {}
        for decision in self._audit_log:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
            if decision.assignment is not None:
                assignment_type = decision.assignment.assignment_type
                assignment_counts[assignment_type] = assignment_counts.get(assignment_type, 0) + 1
                for domain_id in decision.assignment.domain_ids:
                    domain_decisions[domain_id] = domain_decisions.get(domain_id, 0) + 1
        domain_coverage = self._org_chart.entity_coverage()
        return {
            "entities": len(self._kg.entities),
            "triples": len(self._kg.triples),
            "domains": len(self._org_chart.domains),
            "cross_domain_relations": len(self._org_chart.cross_domain_relations),
            "governance_mode": self._governance_mode,
            "audit_log_entries": len(self._audit_log),
            "pending_review": len(self._pending_review),
            "decision_counts": action_counts,
            "assignment_counts": assignment_counts,
            "domain_decision_counts": domain_decisions,
            "domain_coverage": {
                **domain_coverage,
                "fraction_entities_assigned": round(
                    domain_coverage.get("entities_with_domains", 0) / max(len(self._kg.entities), 1),
                    4,
                ),
            },
            "bootstrap_assignment_stats": self._bootstrap_assignment_stats,
            "triage_stats": self._triage_stats,
            "governance_policy": {
                "min_admission_confidence": self._min_admission_confidence,
                "confidence_policy_label": self._confidence_policy_label,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_graph": self._kg.to_dict(),
            "org_chart": self._org_chart.to_dict(),
            "governance_mode": self._governance_mode,
            "audit_log": [decision.to_dict() for decision in self._audit_log],
            "bootstrap_assignment_stats": self._bootstrap_assignment_stats,
            "triage_stats": self._triage_stats,
            "governance_policy": {
                "min_admission_confidence": self._min_admission_confidence,
                "confidence_policy_label": self._confidence_policy_label,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernedKnowledgeGraph":
        if "knowledge_graph" not in data:
            kg = KnowledgeGraph.from_dict(data)
            return cls(kg=kg, governance_mode="audit_only")

        kg = KnowledgeGraph.from_dict(data["knowledge_graph"])
        org_chart_data = data.get("org_chart", {})
        org_chart = OrgChart.from_dict(org_chart_data, kg) if org_chart_data else OrgChart()
        graph = cls(
            kg=kg,
            org_chart=org_chart,
            governance_mode=data.get("governance_mode", "audit_only"),
            min_admission_confidence=(
                data.get("governance_policy", {}).get("min_admission_confidence")
            ),
            confidence_policy_label=(
                data.get("governance_policy", {}).get("confidence_policy_label")
                or "confidence_policy"
            ),
        )
        graph._audit_log = [
            GovernanceDecision.from_dict(item)
            for item in data.get("audit_log", [])
        ]
        graph._bootstrap_assignment_stats = data.get("bootstrap_assignment_stats", {})
        graph._triage_stats = data.get(
            "triage_stats",
            {
                "auto_approved_low_risk": 0,
                "reviewed": 0,
                "policy_rejected": 0,
                "escalated_without_callback": 0,
                "review_reasons": {},
            },
        )
        return graph


__all__ = ["GovernanceDecision", "GovernedKnowledgeGraph"]
