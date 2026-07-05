"""Stage-9 entity-funnel fixes (HYBRID_V2_RUN.md change #1) and wide-harvest
relation seeding (change #2).

The audit found three silent leaks at KG integration: a `_\\d+$` id-cleanup
that ate 4-digit year suffixes (collapsing distinct events), same-id collisions
discarding the second entity's aliases, and wide-harvest relationship
candidates stashed but never consumed.
"""
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


def _organizer() -> KnowledgeOrganizer:
    org = KnowledgeOrganizer.__new__(KnowledgeOrganizer)
    org.knowledge_graph = KnowledgeGraph()
    org.governed_kg = None
    org.shared_memory = None
    org._embeddings_available = False
    org._allowed_relation_types = lambda: set()
    org.integration_stats = {}
    return org


def test_year_suffix_ids_not_collapsed() -> None:
    # "_1998" / "_2002" are identities, not extractor artifacts.
    org = _organizer()
    org._integrate_to_kg(
        entities=[
            {"id": "world_championships_1998", "text": "1998 World Championships", "type": "EVENT"},
            {"id": "world_championships_2002", "text": "2002 World Championships", "type": "EVENT"},
        ],
        triples=[],
        document_id="doc",
    )
    ids = set(org.knowledge_graph.entities)
    assert "world_championships_1998" in ids and "world_championships_2002" in ids


def test_artifact_suffix_still_stripped() -> None:
    org = _organizer()
    org._integrate_to_kg(
        entities=[{"id": "alice_chen_2", "text": "Alice Chen", "type": "PERSON"}],
        triples=[],
        document_id="doc",
    )
    assert "alice_chen" in org.knowledge_graph.entities
    assert "alice_chen_2" not in org.knowledge_graph.entities


def test_collision_merges_aliases_instead_of_discarding() -> None:
    org = _organizer()
    org._integrate_to_kg(
        entities=[
            {"id": "fibt_world_championships", "text": "FIBT World Championships", "type": "EVENT"},
            {"id": "fibt_world_championships", "text": "World Championships", "type": "EVENT",
             "mentions": ["the championships"]},
        ],
        triples=[],
        document_id="doc",
    )
    entity = org.knowledge_graph.entities["fibt_world_championships"]
    assert "FIBT World Championships" in entity.labels
    assert "World Championships" in entity.labels          # was silently discarded
    assert "the championships" in entity.labels


def test_unreferenced_year_kept_bare_number_dropped() -> None:
    # A year with no triple attached is still a real entity (memory node);
    # a bare count with no triple is noise. Domain-general value-shape rule.
    org = _organizer()
    org._integrate_to_kg(
        entities=[
            {"id": "1869", "text": "1869", "type": "UNKNOWN"},   # unreferenced year -> keep as DATE
            {"id": "91", "text": "91", "type": "UNKNOWN"},       # unreferenced bare number -> drop
            {"id": "kyoto", "text": "Kyoto", "type": "LOC"},
        ],
        triples=[],
        document_id="doc",
    )
    ids = set(org.knowledge_graph.entities)
    assert "1869" in ids and "kyoto" in ids
    assert "91" not in ids
    assert org.knowledge_graph.entities["1869"].type == "DATE"


def test_seed_candidates_merged_and_deduped() -> None:
    rex = RelationExtractor.__new__(RelationExtractor)
    all_triples = [{"subject": "alice", "relation": "WORKS_AT", "object": "acme", "confidence": 0.8}]
    seeds = [
        {"source": "Alice", "relation": "works at", "target": "Acme"},     # dup of existing
        {"source": "alice", "relation": "founded in", "target": "2011"},   # new
        {"source": "x", "relation": "", "target": "y"},                    # malformed
        {"source": "z", "relation": "SELF", "target": "z"},                # self-loop
    ]
    added = rex._merge_seed_candidates(all_triples, seeds, entities=[])
    assert added == 1
    seeded = all_triples[-1]
    assert seeded["relation"] == "FOUNDED_IN"
    assert seeded["metadata"]["seeded_from_wide"] is True
    assert len(all_triples) == 2
