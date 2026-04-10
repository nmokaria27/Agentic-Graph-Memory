from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.core import Domain, GovernedKnowledgeGraph, KnowledgeGraph, LLMConfig, OrgChart


def _governed_graph() -> GovernedKnowledgeGraph:
    kg = KnowledgeGraph()
    domain = Domain(
        domain_id="science",
        label="Science",
        description="Scientific domain",
        entity_ids=set(),
        relation_schema={"Used-for": "", "Compare": ""},
        metadata={"seed_relation_types": ["Used-for", "Compare"]},
    )
    return GovernedKnowledgeGraph(kg=kg, org_chart=OrgChart(domains=[domain]), governance_mode="audit_only")


def test_schema_enforcement_maps_canonical_relation() -> None:
    organizer = KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=_governed_graph(),
        llm_config=LLMConfig(model="test-model"),
    )

    relation, allowed = organizer._enforce_relation_schema("USED-FOR", {"Used-for", "Compare"})

    assert allowed is True
    assert relation == "Used-for"


def test_schema_enforcement_rejects_unknown_relation() -> None:
    organizer = KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=_governed_graph(),
        llm_config=LLMConfig(model="test-model"),
    )

    relation, allowed = organizer._enforce_relation_schema("CAUSES", {"Used-for", "Compare"})

    assert allowed is False
    assert relation == "CAUSES"


def test_semantic_duplicate_finder_collapses_near_duplicates() -> None:
    organizer = KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=_governed_graph(),
        llm_config=LLMConfig(model="test-model"),
    )
    merges, remaining = organizer._find_semantic_duplicates(
        [
            {"id": "named_entity_recognition", "text": "Named Entity Recognition", "type": "TASK"},
            {"id": "named_entity_recognition_v2", "text": "Named-Entity Recognition", "type": "TASK"},
            {"id": "machine_translation", "text": "Machine Translation", "type": "TASK"},
        ]
    )

    assert len(merges) == 1
    assert len(remaining) == 2
