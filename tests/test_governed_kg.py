from multi_agent_kg.core import (
    DeliberativeOrchestrator,
    Domain,
    GovernedKnowledgeGraph,
    GovernanceAssignment,
    GovernanceDecision,
    KnowledgeGraph,
    LLMConfig,
    OrgChart,
    Triple,
)
from multi_agent_kg.core.domain_builder import DomainBuilder


def _simple_org_chart() -> OrgChart:
    cardio = Domain(
        domain_id="cardio",
        label="Cardio",
        description="Cardiovascular knowledge",
        entity_ids={"heart", "blood_pressure"},
        relation_schema={"AFFECTS": ""},
        metadata={"seed_entity_types": ["ORGAN", "MEASUREMENT"], "seed_relation_types": ["AFFECTS"]},
    )
    inflam = Domain(
        domain_id="inflammation",
        label="Inflammation",
        description="Inflammatory knowledge",
        entity_ids={"il6", "crp"},
        relation_schema={"ELEVATES": ""},
        metadata={"seed_entity_types": ["MARKER"], "seed_relation_types": ["ELEVATES"]},
    )
    return OrgChart(domains=[cardio, inflam], cross_domain_relations=[])


def test_strict_mode_blocks_unapproved_triples() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("il6", ["IL-6"], "MARKER")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="strict")

    decision = gkg.propose_triple("heart", "AFFECTS", "il6", confidence=0.8)

    assert decision.action == "escalate"
    assert len(gkg.triples) == 0
    assert len(gkg.audit_log) == 1


def test_audit_only_passes_everything() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="audit_only")

    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)

    assert decision.action == "auto_approve"
    assert decision.committed is True
    assert len(gkg.triples) == 1
    assert gkg.triples[0].relation == "AFFECTS"


def test_triage_auto_approves_low_risk_triples_without_callback() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="triage")

    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)

    assert decision.action == "auto_approve"
    assert decision.committed is True
    assert len(gkg.triples) == 1


def test_triage_reviews_low_confidence_triples_when_callback_available() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="triage")

    callback_calls = {"count": 0}

    def callback(triple, assignment, _kg, _org_chart):
        callback_calls["count"] += 1
        return GovernanceDecision(
            triple=triple,
            action="approve",
            domain_id=assignment.primary_domain_id,
            rationale="approved after triage review",
            assignment=assignment,
        )

    gkg.set_review_callback(callback)
    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.4)

    assert callback_calls["count"] == 1
    assert decision.action == "approve"
    assert decision.committed is True
    assert "triage_reason=low_confidence" in decision.rationale


def test_triage_confidence_policy_rejects_before_review() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(
        kg=kg,
        org_chart=_simple_org_chart(),
        governance_mode="triage",
        min_admission_confidence=0.75,
        confidence_policy_label="fixed_schema_confidence_floor",
    )

    callback_calls = {"count": 0}

    def callback(triple, assignment, _kg, _org_chart):
        callback_calls["count"] += 1
        return GovernanceDecision(
            triple=triple,
            action="approve",
            domain_id=assignment.primary_domain_id,
            rationale="should not be called",
            assignment=assignment,
        )

    gkg.set_review_callback(callback)
    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.74)

    assert callback_calls["count"] == 0
    assert decision.action == "reject"
    assert decision.committed is False
    assert len(gkg.triples) == 0
    assert len(gkg.audit_log) == 1
    assert "fixed_schema_confidence_floor" in decision.rationale
    stats = gkg.get_stats()
    assert stats["decision_counts"]["reject"] == 1
    assert stats["triage_stats"]["policy_rejected"] == 1
    assert stats["triage_stats"]["review_reasons"]["fixed_schema_confidence_floor"] == 1


def test_triage_reviews_conflicts_when_callback_available() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    kg.add_entity("artery", ["Artery"], "ORGAN")
    kg.add_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="triage")
    gkg.assign_entity_to_domains("artery", ["cardio"])

    callback_calls = {"count": 0}

    def callback(triple, assignment, _kg, _org_chart):
        callback_calls["count"] += 1
        return GovernanceDecision(
            triple=triple,
            action="reject",
            domain_id=assignment.primary_domain_id,
            rationale="rejected conflicting triple",
            assignment=assignment,
        )

    gkg.set_review_callback(callback)
    decision = gkg.propose_triple("heart", "AFFECTS", "artery", confidence=0.95)

    assert callback_calls["count"] == 1
    assert decision.action == "reject"
    assert decision.committed is False
    assert "triage_reason=conflict" in decision.rationale


def test_serialization_round_trip() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="audit_only")
    gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)

    restored = GovernedKnowledgeGraph.from_dict(gkg.to_dict())

    assert restored.governance_mode == "audit_only"
    assert len(restored.triples) == 1
    assert len(restored.org_chart.domains) == 2
    assert len(restored.audit_log) == 1


def test_backward_compat_loads_plain_kg() -> None:
    plain = {
        "entities": [
            {"id": "heart", "labels": ["Heart"], "type": "ORGAN", "metadata": {}},
            {"id": "blood_pressure", "labels": ["Blood Pressure"], "type": "MEASUREMENT", "metadata": {}},
        ],
        "triples": [
            {
                "subject": "heart",
                "relation": "AFFECTS",
                "object": "blood_pressure",
                "confidence": 0.8,
                "source": "doc1",
                "metadata": {},
            }
        ],
    }
    restored = GovernedKnowledgeGraph.from_dict(plain)

    assert restored.governance_mode == "audit_only"
    assert len(restored.entities) == 2
    assert len(restored.triples) == 1


def test_propose_triple_routes_correctly() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="permissive")

    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.7)

    assert decision.assignment is not None
    assert decision.assignment.assignment_type == "single_owner"
    assert decision.assignment.primary_domain_id == "cardio"


def test_audit_log_records_all_decisions() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="permissive")

    gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.7)
    gkg.add_triple_bypass("heart", "TRACKS", "blood_pressure", confidence=0.6)

    assert len(gkg.audit_log) == 2
    assert {decision.action for decision in gkg.audit_log} == {"approve", "auto_approve"}


def test_bootstrap_domains() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("cnn", ["CNN"], "METHOD")
    kg.add_entity("translation", ["Machine Translation"], "TASK")
    kg.add_triple("cnn", "USED_FOR", "translation", confidence=0.9)
    gkg = GovernedKnowledgeGraph(kg=kg, governance_mode="audit_only")
    builder = DomainBuilder(LLMConfig(model="test-model"), target_num_domains=1)

    org_chart = gkg.bootstrap_domains(builder)

    assert len(org_chart.domains) == 1
    assert org_chart.domains[0].domain_id == "global_expert"


def test_assign_entity_to_domains_updates_org_chart() -> None:
    gkg = GovernedKnowledgeGraph(kg=KnowledgeGraph(), org_chart=_simple_org_chart(), governance_mode="audit_only")
    gkg.add_entity("artery", ["Artery"], "ORGAN")
    gkg.assign_entity_to_domains("artery", ["cardio"])

    assert "artery" in gkg.org_chart.find_domain("cardio").entity_ids


def test_strict_mode_uses_review_callback_when_available() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="strict")

    def callback(triple, assignment, _kg, _org_chart):
        return GovernanceDecision(
            triple=triple,
            action="approve",
            domain_id=assignment.primary_domain_id,
            rationale="approved in test",
            assignment=assignment,
        )

    gkg.set_review_callback(callback)
    decision = gkg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)

    assert decision.action == "approve"
    assert decision.committed is True
    assert len(gkg.triples) == 1


def test_cross_domain_relations_update_incrementally_on_commit() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("il6", ["IL-6"], "MARKER")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="audit_only")

    decision = gkg.propose_triple("heart", "AFFECTS", "il6", confidence=0.8)

    assert decision.committed is True
    assert len(gkg.org_chart.cross_domain_relations) == 1


def test_bootstrap_assignment_stats_are_recorded() -> None:
    gkg = GovernedKnowledgeGraph(kg=KnowledgeGraph(), org_chart=_simple_org_chart(), governance_mode="audit_only")
    stats = {
        "num_entities": 4,
        "assigned_entities": 3,
        "unassigned_entities": 1,
        "multi_assigned_entities": 1,
        "assignment_coverage": 0.75,
    }
    gkg.set_bootstrap_assignment_stats(stats)

    stored = gkg.get_stats()["bootstrap_assignment_stats"]

    assert stored["assigned_entities"] == 3
    assert stored["assignment_coverage"] == 0.75


def test_strict_mode_reuses_one_review_board(monkeypatch) -> None:
    from multi_agent_kg.core import incremental_enrichment

    init_calls = {"count": 0}

    class DummyBoard:
        def __init__(self, org_chart, base_kg, llm_config):
            init_calls["count"] += 1
            self.org_chart = org_chart
            self.base_kg = base_kg
            self.llm_config = llm_config

        def _review_candidate(self, candidate, assignment, source_text="", **_kwargs):
            return {
                "action": "approve",
                "rationale": "approved by dummy board",
                "revised_triple": None,
            }

    monkeypatch.setattr(incremental_enrichment, "GovernanceReviewBoard", DummyBoard)

    kg = KnowledgeGraph()
    kg.add_entity("heart", ["Heart"], "ORGAN")
    kg.add_entity("blood_pressure", ["Blood Pressure"], "MEASUREMENT")
    kg.add_entity("il6", ["IL-6"], "MARKER")
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=_simple_org_chart(), governance_mode="strict")

    orchestrator = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        governed_kg=gkg,
        governance_mode="strict",
        enable_deliberation=False,
        enable_self_consistency=False,
        enable_open_world=False,
        enable_cross_document=False,
    )

    decision_one = orchestrator.governed_kg.propose_triple("heart", "AFFECTS", "blood_pressure", confidence=0.9)
    decision_two = orchestrator.governed_kg.propose_triple("heart", "AFFECTS", "il6", confidence=0.8)

    assert init_calls["count"] == 1
    assert decision_one.committed is True
    assert decision_two.committed is True
