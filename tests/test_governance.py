from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.domain_experts import Domain, DomainBuilder, OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple


def _build_test_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity("insulin_resistance", ["Insulin Resistance"], "CONDITION")
    kg.add_entity("endothelial_dysfunction", ["Endothelial Dysfunction"], "CONDITION")
    kg.add_entity("il6", ["IL-6"], "BIOMARKER")
    kg.add_entity("c_reactive_protein", ["C-reactive protein"], "BIOMARKER")
    kg.add_triple("insulin_resistance", "ASSOCIATED_WITH", "endothelial_dysfunction", 0.9)
    kg.add_triple("il6", "ELEVATED_IN", "insulin_resistance", 0.8)
    return kg


def test_route_triple_for_governance_single_owner() -> None:
    cardio = Domain(
        domain_id="cardio",
        label="Cardiovascular",
        description="Cardiovascular mechanisms",
        entity_ids={"insulin_resistance", "endothelial_dysfunction"},
        relation_schema={"ASSOCIATED_WITH": ""},
        metadata={"owner_label": "Cardio Expert"},
    )
    inflam = Domain(
        domain_id="inflammation",
        label="Inflammation",
        description="Inflammatory markers",
        entity_ids={"il6", "c_reactive_protein"},
        relation_schema={"ELEVATED_IN": ""},
        metadata={"owner_label": "Inflammation Expert"},
    )
    org_chart = OrgChart(domains=[cardio, inflam], cross_domain_relations=[])

    assignment = org_chart.route_triple_for_governance(
        Triple("insulin_resistance", "ASSOCIATED_WITH", "endothelial_dysfunction")
    )

    assert assignment.assignment_type == "single_owner"
    assert assignment.primary_domain_id == "cardio"
    assert assignment.domain_ids == ["cardio"]


def test_route_triple_for_governance_cross_domain() -> None:
    cardio = Domain(
        domain_id="cardio",
        label="Cardiovascular",
        description="Cardiovascular mechanisms",
        entity_ids={"insulin_resistance"},
        relation_schema={"ASSOCIATED_WITH": ""},
    )
    inflam = Domain(
        domain_id="inflammation",
        label="Inflammation",
        description="Inflammatory markers",
        entity_ids={"il6"},
        relation_schema={"ASSOCIATED_WITH": ""},
    )
    org_chart = OrgChart(domains=[cardio, inflam], cross_domain_relations=[])

    assignment = org_chart.route_triple_for_governance(
        Triple("insulin_resistance", "ASSOCIATED_WITH", "il6")
    )

    assert assignment.assignment_type == "cross_domain"
    assert assignment.domain_ids == ["cardio", "inflammation"]


def test_org_chart_round_trip_preserves_domain_metadata() -> None:
    kg = _build_test_kg()
    domain = Domain(
        domain_id="cardio",
        label="Cardiovascular",
        description="Cardiovascular mechanisms",
        entity_ids={"insulin_resistance", "endothelial_dysfunction"},
        relation_schema={"ASSOCIATED_WITH": ""},
        metadata={
            "owner_label": "Cardio Expert",
            "governance_scope": "Owns metabolic-vascular facts.",
        },
    )
    org_chart = OrgChart(domains=[domain], cross_domain_relations=[])

    restored = OrgChart.from_dict(org_chart.to_dict(), kg)

    assert restored.domains[0].owner_label == "Cardio Expert"
    assert restored.domains[0].governance_scope == "Owns metabolic-vascular facts."


def test_domain_builder_single_domain_mode() -> None:
    kg = _build_test_kg()
    builder = DomainBuilder(LLMConfig(model="test-model"), target_num_domains=1)

    org_chart = builder.build(kg)

    assert len(org_chart.domains) == 1
    assert org_chart.domains[0].domain_id == "global_expert"
    assert org_chart.domains[0].entity_ids == set(kg.entities.keys())
