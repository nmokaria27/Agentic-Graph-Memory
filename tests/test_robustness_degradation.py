"""EXP-3 fault-injection tests — G0: stage failures must degrade, never
destroy completed extraction work.

Trigger B (observed twice on LongMemEval smoke runs): the stage-9 entity
dedup LLM returned non-dict elements inside merge_groups; calling .get on a
str crashed stage 9 and the whole document's KG was forfeited.
"""
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.core import Domain, GovernedKnowledgeGraph, KnowledgeGraph, LLMConfig, OrgChart


def _organizer() -> KnowledgeOrganizer:
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
    return KnowledgeOrganizer(
        knowledge_graph=KnowledgeGraph(),
        governed_kg=governed,
        llm_config=LLMConfig(model="test-model"),
    )


# Distinct names so neither the obvious nor the semantic (lexical-similarity)
# pre-dedup collapses them — the LLM dedup path must actually run.
_ENTITIES = [
    {"id": "marie_curie", "text": "Marie Curie", "type": "Person"},
    {"id": "albert_einstein", "text": "Albert Einstein", "type": "Person"},
    {"id": "warsaw", "text": "Warsaw", "type": "Location"},
    {"id": "nobel_prize", "text": "Nobel Prize", "type": "Award"},
]


def test_dedup_skips_malformed_merge_groups(monkeypatch) -> None:
    """Non-dict elements in merge_groups are skipped; valid same-type ones still apply.

    The merge target is a same-type (Person) but lexically/semantically
    unrelated entity ("Albert Einstein"), not "warsaw" (Location) — a
    cross-type merge is exactly the GB-8 pathology the dedup guard blocks
    (see test_dedup_guard_blocks_cross_type_merge below). Deliberately
    dissimilar names avoid the pre-LLM obvious/semantic dedup (trigram +
    live embedding similarity) collapsing them before the LLM merge path
    under test ever runs.
    """
    organizer = _organizer()
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                "garbage string the model emitted",   # crashed stage 9 pre-fix
                42,
                {"canonical_id": "marie_curie", "canonical_name": "Marie Curie",
                 "merge_ids": ["albert_einstein"]},
            ]
        },
    )

    remaining, merged_count = organizer._deduplicate_entities(list(_ENTITIES))

    ids = {e["id"] for e in remaining}
    assert "albert_einstein" not in ids   # the one valid same-type merge applied
    assert "marie_curie" in ids
    assert "warsaw" in ids                       # untouched by this merge
    assert merged_count == 1


def test_dedup_guard_blocks_cross_type_merge(monkeypatch) -> None:
    """GB-8: a cross-type merge (Location into Person) must be rejected."""
    organizer = _organizer()
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                {"canonical_id": "marie_curie", "canonical_name": "Marie Curie",
                 "merge_ids": ["warsaw"]},
            ]
        },
    )

    remaining, merged_count = organizer._deduplicate_entities(list(_ENTITIES))

    ids = {e["id"] for e in remaining}
    assert "warsaw" in ids              # cross-type merge blocked, nothing deleted
    assert "marie_curie" in ids
    assert merged_count == 0


def test_dedup_guard_blocks_mega_merge(monkeypatch) -> None:
    """GB-8: a merge group swallowing most of the document must be rejected.

    Reproduces the EXP-MODEL-LOCAL collapse: Qwen3 hybrid slice A doc_0 kept
    2 entities (both DATE) with 59 triples after the LLM proposed merging
    every other entity into one canonical, regardless of type.
    """
    entities = [
        {"id": "wilfried_schneider", "text": "Wilfried Schneider", "type": "Person"},
        {"id": "turin", "text": "Turin", "type": "Location"},
        {"id": "german", "text": "German", "type": "Nationality"},
        {"id": "british_columbia", "text": "British Columbia", "type": "Location"},
        {"id": "skeleton_racer", "text": "skeleton racer", "type": "Occupation"},
        {"id": "melissa_hollingsworth", "text": "Melissa Hollingsworth", "type": "Person"},
        {"id": "1997", "text": "1997", "type": "Date"},
        {"id": "2002", "text": "2002", "type": "Date"},
        {"id": "13_march_1963", "text": "13 March 1963", "type": "Date"},
        {"id": "july_2012", "text": "July 2012", "type": "Date"},
    ]
    organizer = _organizer()
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [{
                "canonical_id": "13_march_1963",
                "canonical_name": "13 March 1963",
                "merge_ids": [e["id"] for e in entities if e["id"] != "13_march_1963"],
            }]
        },
    )

    remaining, merged_count = organizer._deduplicate_entities(list(entities))

    ids = {e["id"] for e in remaining}
    non_dates = {e["id"] for e in entities if e["type"] != "Date"}
    assert non_dates.issubset(ids), f"cross-type entities deleted: {non_dates - ids}"
    assert len(remaining) >= 7, f"catastrophic collapse not prevented: kept {len(remaining)}/10"


def test_dedup_guard_preserves_aliases_on_valid_merge(monkeypatch) -> None:
    """GB-8: a merged entity's surface form survives as an alias on the canonical."""
    organizer = _organizer()
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                {"canonical_id": "marie_curie", "canonical_name": "Marie Curie",
                 "merge_ids": ["albert_einstein"]},
            ]
        },
    )

    remaining, merged_count = organizer._deduplicate_entities(list(_ENTITIES))

    canonical = next(e for e in remaining if e["id"] == "marie_curie")
    assert "Albert Einstein" in canonical.get("labels", [])
    assert merged_count == 1


def test_dedup_llm_failure_degrades_to_unmerged(monkeypatch) -> None:
    """An LLM API failure in dedup keeps all entities instead of raising."""
    organizer = _organizer()

    def _boom(**kwargs):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr(organizer, "call_llm", _boom)

    remaining, merged_count = organizer._deduplicate_entities(list(_ENTITIES))

    assert {e["id"] for e in remaining} == {e["id"] for e in _ENTITIES}
    assert merged_count == 0
