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


def test_org_chart_from_dict_refreshes_domain_memory_cards() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("bert", ["BERT"], "Method")
    kg.add_entity("classification", ["classification"], "Task")
    kg.add_triple(
        "bert",
        "Used-for",
        "classification",
        0.9,
        source="doc1",
        metadata={"evidence": "BERT is used for classification."},
    )
    org_chart = OrgChart(
        domains=[
            Domain(
                domain_id="methods",
                label="Methods",
                description="Methods and tasks",
                entity_ids={"bert", "classification"},
                relation_schema={"Used-for": "method usage"},
            )
        ],
        cross_domain_relations=[],
    )

    restored = OrgChart.from_dict(org_chart.to_dict(), kg)

    assert "memory_card" in restored.domains[0].metadata
    assert "BERT is used for classification" in restored.domains[0].memory_card_summary()


def test_domain_memory_card_summarizes_owned_graph() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("bert", ["BERT"], "Method")
    kg.add_entity("classification", ["classification"], "Task")
    kg.add_triple("bert", "Used-for", "classification", 0.9, source="doc1")
    domain = Domain(
        domain_id="methods",
        label="Methods",
        description="Methods and tasks",
        entity_ids={"bert", "classification"},
        relation_schema={"Used-for": "method usage"},
        topics=[],
    )

    card = domain.refresh_memory_card(kg)

    assert card["domain_id"] == "methods"
    assert "BERT" in card["key_entities"]
    assert "(bert) -[Used-for]-> (classification)" in card["core_facts"][0]
    assert domain.metadata["memory_card"]["triple_count"] == 1
    assert "DOMAIN MEMORY CARD" in domain.memory_card_summary()


def test_domain_memory_card_includes_evidence_snippets() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("bert", ["BERT"], "Method")
    kg.add_entity("classification", ["classification"], "Task")
    kg.add_triple(
        "bert",
        "Used-for",
        "classification",
        0.9,
        source="doc1",
        metadata={"evidence": "BERT is used for text classification in the source abstract."},
    )
    domain = Domain(
        domain_id="methods",
        label="Methods",
        description="Methods and tasks",
        entity_ids={"bert", "classification"},
        relation_schema={"Used-for": "method usage"},
        topics=[],
    )

    card = domain.refresh_memory_card(kg)
    summary = domain.memory_card_summary()

    assert "BERT is used for text classification" in card["evidence_snippets"][0]
    assert "Evidence snippets" in summary
    assert "BERT is used for text classification" in summary


def test_domain_builder_single_domain_mode() -> None:
    kg = _build_test_kg()
    builder = DomainBuilder(LLMConfig(model="test-model"), target_num_domains=1)

    org_chart = builder.build(kg)

    assert len(org_chart.domains) == 1
    assert org_chart.domains[0].domain_id == "global_expert"
    assert org_chart.domains[0].entity_ids == set(kg.entities.keys())
