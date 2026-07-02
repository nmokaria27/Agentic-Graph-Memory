"""Tests for the persistent entity-alias table on GovernedKnowledgeGraph."""

from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph


def _kg_with_entities():
    gkg = GovernedKnowledgeGraph(governance_mode="audit_only")
    gkg.add_entity("marie_curie", labels=["Marie Curie"], entity_type="PERSON")
    gkg.add_entity("radium_institute", labels=["Radium Institute"], entity_type="ORG")
    return gkg


def test_register_alias_basic():
    gkg = _kg_with_entities()
    gkg.register_alias("M. Curie", "marie_curie")
    assert gkg.resolve_alias("M. Curie") == "marie_curie"
    assert gkg.resolve_alias("unknown") == "unknown"


def test_register_alias_ignores_self_and_empty():
    gkg = _kg_with_entities()
    gkg.register_alias("marie_curie", "marie_curie")
    gkg.register_alias("", "marie_curie")
    gkg.register_alias("x", "")
    assert gkg.entity_aliases == {}


def test_register_alias_flattens_chains():
    gkg = _kg_with_entities()
    gkg.register_alias("curie", "marie_curie")
    # "curie" is itself an alias; new alias should point at the final target
    gkg.register_alias("mme_curie", "curie")
    assert gkg.entity_aliases["mme_curie"] == "marie_curie"


def test_register_alias_rejects_two_cycle():
    gkg = _kg_with_entities()
    gkg.register_alias("a", "b")
    gkg.register_alias("b", "a")  # would create a->b->a
    assert gkg.resolve_alias("a") == "b"
    assert "b" not in gkg.entity_aliases


def test_sync_aliases_from_returns_added_count():
    gkg = _kg_with_entities()
    added = gkg.sync_aliases_from({"MC": "marie_curie", "RI": "radium_institute"})
    assert added == 2
    added_again = gkg.sync_aliases_from({"MC": "marie_curie"})
    assert added_again == 0


def test_aliases_survive_serialization_roundtrip():
    gkg = _kg_with_entities()
    gkg.register_alias("M. Curie", "marie_curie")
    gkg.register_alias("the institute", "radium_institute")

    restored = GovernedKnowledgeGraph.from_dict(gkg.to_dict())

    assert restored.entity_aliases == {
        "M. Curie": "marie_curie",
        "the institute": "radium_institute",
    }
    assert restored.resolve_alias("M. Curie") == "marie_curie"
