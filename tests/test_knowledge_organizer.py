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


def test_integrate_to_kg_skips_self_loops_and_duplicate_triples() -> None:
    governed = _governed_graph()
    organizer = KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=governed,
        llm_config=LLMConfig(model="test-model"),
    )

    entities = [
        {"id": "method_a", "text": "Method A", "type": "METHOD"},
        {"id": "task_b", "text": "Task B", "type": "TASK"},
    ]
    triples = [
        {"subject": "Method A", "relation": "Used-for", "object": "Task B", "confidence": 0.9},
        {"subject": "Method A", "relation": "Used-for", "object": "Task B", "confidence": 0.9},
        {"subject": "Method A", "relation": "Part-of", "object": "Method A", "confidence": 0.9},
    ]

    added_entities, added_triples = organizer._integrate_to_kg(
        entities=entities,
        triples=triples,
        document_id="doc1",
    )

    assert added_entities == 2
    assert added_triples == 1
    assert len(governed.kg.triples) == 1
    triple = governed.kg.triples[0]
    assert triple.subject == "method_a"
    assert triple.object == "task_b"
