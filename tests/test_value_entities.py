"""Stage-9 value-entity policy (MATRIX_REPORT.md recommendation #1).

The integration garbage filter must not delete year/number entities that a
fact references, and value-shaped triple endpoints must materialize as typed
nodes instead of killing the triple (skip reason 'unresolved_entity').
"""
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer, classify_value
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def _organizer():
    return KnowledgeOrganizer(knowledge_graph=KnowledgeGraph())


def test_classify_value_shapes():
    assert classify_value("1997") == "DATE"
    assert classify_value("1997-98") == "DATE"
    assert classify_value("13 March 1963") == "DATE"
    assert classify_value("September 1911") == "DATE"
    assert classify_value("12,000") == "NUMBER"
    assert classify_value("91") == "NUMBER"
    assert classify_value("23 years") == "QUANTITY"
    assert classify_value("70 percent") == "QUANTITY"
    assert classify_value("Willi Schneider") is None
    assert classify_value("") is None
    assert classify_value("a phrase far too long to ever be a value node here") is None


def test_referenced_year_entity_is_kept_and_typed():
    org = _organizer()
    entities = [
        {"id": "schneider", "text": "Schneider", "type": "PERSON", "confidence": 0.9},
        {"id": "1997", "text": "1997", "type": "UNKNOWN", "confidence": 0.8},
        {"id": "42", "text": "42", "type": "UNKNOWN", "confidence": 0.8},  # unreferenced
    ]
    triples = [
        {"subject": "Schneider", "relation": "WON_TITLE_IN", "object": "1997", "confidence": 0.8},
    ]
    added_e, added_t = org._integrate_to_kg(entities, triples, "doc1")
    kg = org.knowledge_graph
    assert "1997" in kg.entities            # referenced year survives
    assert kg.entities["1997"].type == "DATE"
    assert "42" not in kg.entities          # unreferenced stray number still dropped
    assert added_t == 1                     # the year triple lands


def test_value_endpoint_materializes_instead_of_skipping_triple():
    org = _organizer()
    entities = [{"id": "schneider", "text": "Schneider", "type": "PERSON", "confidence": 0.9}]
    triples = [
        {"subject": "Schneider", "relation": "BORN_ON", "object": "13 March 1963", "confidence": 0.9},
        {"subject": "Schneider", "relation": "KNOWS", "object": "Completely Unknown Person", "confidence": 0.5},
    ]
    added_e, added_t = org._integrate_to_kg(entities, triples, "doc1")
    kg = org.knowledge_graph
    assert added_t == 1                     # date triple rescued, unknown-person one skipped
    assert "13_march_1963" in kg.entities
    assert kg.entities["13_march_1963"].type == "DATE"
    assert not any("completely_unknown" in eid for eid in kg.entities)


def test_named_entities_unaffected_by_value_policy():
    org = _organizer()
    entities = [
        {"id": "acme", "text": "Acme Corp", "type": "ORG", "confidence": 0.9},
        {"id": "bob", "text": "Bob", "type": "PERSON", "confidence": 0.9},
    ]
    triples = [{"subject": "Bob", "relation": "WORKS_AT", "object": "Acme Corp", "confidence": 0.9}]
    added_e, added_t = org._integrate_to_kg(entities, triples, "doc1")
    assert added_e == 2 and added_t == 1
