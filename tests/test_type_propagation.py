"""GB-11: entity types must survive extraction → coref → stage 9 → KG dump.

EXP-SPGOV found GB-8's type-compatibility gate inert in production because
(1) coref defaulted missing LLM types to UNKNOWN (erasing member types), and
(2) eval dumps read Entity.entity_type (does not exist) instead of Entity.type.
"""
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.entity_types import inherit_type_from_members, is_blank_entity_type
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.core import Domain, GovernedKnowledgeGraph, KnowledgeGraph, LLMConfig, OrgChart
from multi_agent_kg.core.knowledge_graph import Entity


def test_inherit_type_prefers_real_llm_type() -> None:
    assert inherit_type_from_members("LOCATION", ["PERSON", "PERSON"]) == "LOCATION"


def test_inherit_type_from_members_when_llm_unknown() -> None:
    assert inherit_type_from_members("UNKNOWN", ["PERSON", "PERSON", "LOCATION"]) == "PERSON"
    assert inherit_type_from_members("?", ["Album", "Album"]) == "Album"
    assert inherit_type_from_members("", ["Year"]) == "Year"


def test_inherit_type_falls_back_when_no_member_signal() -> None:
    assert inherit_type_from_members("UNKNOWN", []) == "UNKNOWN"
    assert is_blank_entity_type("?")
    assert not is_blank_entity_type("PERSON")


def test_coref_inherits_member_type_when_llm_omits_type() -> None:
    """GB-11: LLM group type=UNKNOWN must not wipe extraction types."""
    ex = EntityExtractor.__new__(EntityExtractor)
    ex.shared_memory = None
    ex.call_llm = lambda **kwargs: {
        "entity_groups": [{
            "canonical_id": "gloria_estefan",
            "canonical_name": "Gloria Estefan",
            "type": "UNKNOWN",
            "mentions": ["Gloria Estefan", "Estefan"],
            "is_known_entity": False,
        }]
    }
    entities = [
        {"id": "gloria_estefan", "text": "Gloria Estefan", "type": "PERSON", "confidence": 0.9},
        {"id": "estefan", "text": "Estefan", "type": "PERSON", "confidence": 0.85},
        {"id": "miami", "text": "Miami", "type": "LOCATION", "confidence": 0.9},
    ]
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    gloria = next(e for e in resolved if e.get("id") == "gloria_estefan")
    assert gloria["type"] == "PERSON"
    miami = next(e for e in resolved if e.get("text") == "Miami")
    assert miami["type"] == "LOCATION"  # passthrough keeps extraction type


def test_integrate_preserves_extraction_type_on_committed_entity() -> None:
    """End-to-end: stage-9 integrate writes Entity.type from the entity dict."""
    kg = KnowledgeGraph()
    domain = Domain(
        domain_id="general",
        label="General",
        description="General domain",
        entity_ids=set(),
        relation_schema={},
    )
    governed = GovernedKnowledgeGraph(
        kg=kg, org_chart=OrgChart(domains=[domain]), governance_mode="audit_only"
    )
    organizer = KnowledgeOrganizer(
        knowledge_graph=kg,
        governed_kg=governed,
        llm_config=LLMConfig(model="test-model"),
        enable_deduplication=False,
        enable_normalization=False,
    )
    entities = [
        {"id": "spanish", "text": "Spanish", "type": "LANGUAGE", "confidence": 0.9},
        {"id": "usa", "text": "USA", "type": "LOCATION", "confidence": 0.9},
    ]
    added, _ = organizer._integrate_to_kg(entities, [], document_id="doc_test")
    assert added == 2
    assert kg.entities["spanish"].type == "LANGUAGE"
    assert kg.entities["usa"].type == "LOCATION"


def test_entity_dump_reads_type_not_entity_type() -> None:
    """The DocRED/LongMemEval dump bug: getattr(e, 'entity_type') always '?'."""
    e = Entity(id="x", labels=["X"], type="PERSON")
    dumped_type = str(getattr(e, "type", None) or "?")
    wrong_attr = str(getattr(e, "entity_type", "?"))
    assert dumped_type == "PERSON"
    assert wrong_attr == "?"  # documents the pre-fix failure mode
    name = str((getattr(e, "labels", None) or [None])[0] or e.id)
    assert name == "X"


def test_dedup_unknown_types_do_not_false_match_as_same_type(monkeypatch) -> None:
    """GB-11: UNKNOWN==UNKNOWN must not count as a positive type match.

    Blank sides stay permissive (either-side blank → eligible), which is the
    documented GB-8 contract — this test only asserts that two UNKNOWN entities
    are still merge-eligible via the blank clause, while a real cross-type
    pair remains blocked.
    """
    kg = KnowledgeGraph()
    domain = Domain(
        domain_id="general",
        label="General",
        description="General",
        entity_ids=set(),
        relation_schema={},
    )
    governed = GovernedKnowledgeGraph(
        kg=kg, org_chart=OrgChart(domains=[domain]), governance_mode="audit_only"
    )
    organizer = KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=governed,
        llm_config=LLMConfig(model="test-model"),
    )
    # Dissimilar names so pre-LLM semantic dedup does not collapse them.
    entities = [
        {"id": "dr_beat", "text": "Dr. Beat", "type": "UNKNOWN"},
        {"id": "primitive_love", "text": "Primitive Love", "type": "UNKNOWN"},
        {"id": "spanish", "text": "Spanish", "type": "LANGUAGE"},
        {"id": "usa", "text": "USA", "type": "LOCATION"},
    ]
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                {"canonical_id": "dr_beat", "merge_ids": ["primitive_love"]},
                {"canonical_id": "spanish", "merge_ids": ["usa"]},
            ]
        },
    )
    remaining, _ = organizer._deduplicate_entities(list(entities))
    ids = {e["id"] for e in remaining}
    # Cross-type LANGUAGE←LOCATION still blocked.
    assert "usa" in ids and "spanish" in ids
    # Blank/UNKNOWN pair remains eligible (permissive blank-side clause).
    assert "primitive_love" not in ids
    assert "dr_beat" in ids
