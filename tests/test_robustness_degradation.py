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
    {"id": "warsaw", "text": "Warsaw", "type": "Location"},
    {"id": "nobel_prize", "text": "Nobel Prize", "type": "Award"},
]


def test_dedup_skips_malformed_merge_groups(monkeypatch) -> None:
    """Non-dict elements in merge_groups are skipped; valid ones still apply."""
    organizer = _organizer()
    monkeypatch.setattr(
        organizer,
        "call_llm",
        lambda **kwargs: {
            "merge_groups": [
                "garbage string the model emitted",   # crashed stage 9 pre-fix
                42,
                {"canonical_id": "marie_curie", "canonical_name": "Marie Curie",
                 "merge_ids": ["warsaw"]},
            ]
        },
    )

    remaining, merged_count = organizer._deduplicate_entities(list(_ENTITIES))

    ids = {e["id"] for e in remaining}
    assert "warsaw" not in ids          # the one valid merge was applied
    assert "marie_curie" in ids
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
