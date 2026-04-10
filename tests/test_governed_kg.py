from multi_agent_kg.core import (
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
