"""GB-12: distinct numeric/date literals must never merge in stage-9 dedup.

EXP-SPGOV-2 traced the residual same-type over-merge to literal absorption
(doc_100: 1911/1992 absorbed by 1939 — all `DATE`, size-legal group, so the
GB-8 type gate is correctly permissive). Exposure exists in both the LLM
merge-group path and the semantic-dedup path ("1984"/"1985" share enough
trigrams to clear the 0.88 cosine bar).
"""
from multi_agent_kg.agents.entity_types import literal_conflict, numeric_signature
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.core import Domain, GovernedKnowledgeGraph, KnowledgeGraph, LLMConfig, OrgChart


def _organizer(**kwargs):
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
    return KnowledgeOrganizer(
        knowledge_graph=kg,
        governed_kg=governed,
        llm_config=LLMConfig(model="test-model"),
        **kwargs,
    )


def test_numeric_signature_normalizes_thousands_separators() -> None:
    assert numeric_signature("1,000 employees") == ("1000",)
    assert numeric_signature("1000") == ("1000",)
    assert numeric_signature("no digits here") == ()


def test_literal_conflict_basic_cases() -> None:
    assert literal_conflict("1911", "1939")
    assert literal_conflict("1 million copies", "100 million")
    assert not literal_conflict("1984", "1984")            # identical literal
    assert not literal_conflict("1,000", "1000")           # separator-normalized
    assert not literal_conflict("1984", "1984 World Tour") # same digit content
    assert not literal_conflict("Gloria Estefan", "Estefan")  # digit-free
    assert not literal_conflict("1939", "World War II")    # one side digit-free


def test_llm_dedup_never_absorbs_distinct_years(monkeypatch) -> None:
    """The doc_100 pathology: 1939 <- [1911, 1992], all DATE (same type, so the
    GB-8 gate permits it) — the literal guard must keep all three."""
    organizer = _organizer()
    entities = [
        {"id": "y1939", "text": "1939", "type": "DATE"},
        {"id": "y1911", "text": "1911", "type": "DATE"},
        {"id": "y1992", "text": "1992", "type": "DATE"},
        {"id": "brightman", "text": "Samuel C. Brightman", "type": "PERSON"},
        {"id": "brightman_alias", "text": "Brightman", "type": "PERSON"},
    ]
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                {"canonical_id": "y1939", "merge_ids": ["y1911", "y1992"]},
                {"canonical_id": "brightman", "merge_ids": ["brightman_alias"]},
            ]
        },
    )
    remaining, merged = organizer._deduplicate_entities(list(entities))
    ids = {e["id"] for e in remaining}
    assert {"y1939", "y1911", "y1992"} <= ids   # every year survives
    assert "brightman_alias" not in ids          # digit-free merge unaffected
    assert merged == 1


def test_semantic_dedup_does_not_merge_adjacent_years() -> None:
    """Trigram cosine of '1984'/'1985' can clear 0.88 — guard must veto."""
    organizer = _organizer()
    organizer._embeddings_available = False
    entities = [
        {"id": "y1984", "text": "1984", "type": "DATE"},
        {"id": "y1985", "text": "1985", "type": "DATE"},
        {"id": "nyc_a", "text": "New York City", "type": "LOCATION"},
        {"id": "nyc_b", "text": "new york city", "type": "LOCATION"},
    ]
    merges, remaining = organizer._find_semantic_duplicates(list(entities))
    merged_ids = {m for g in merges for m in g["merge_ids"]}
    assert "y1985" not in merged_ids and "y1984" not in merged_ids
    assert "nyc_b" in merged_ids  # digit-free near-duplicate still collapses
