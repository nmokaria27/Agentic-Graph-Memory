"""Tests for community detection/summaries and Personalized PageRank retrieval."""

from multi_agent_kg.core.community import (
    build_community_summaries,
    community_context,
    detect_communities,
)
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.graph_traversal import personalized_pagerank


def _two_cluster_kg():
    """Two dense clusters joined by a single bridge edge."""
    gkg = GovernedKnowledgeGraph(governance_mode="audit_only")
    physics = ["curie", "becquerel", "radium", "polonium", "sorbonne"]
    music = ["beatles", "lennon", "mccartney", "abbey_road", "liverpool"]
    for eid in physics + music:
        gkg.add_entity(eid)
    kg = gkg.kg
    for i in range(len(physics) - 1):
        kg.add_triple(physics[i], "RELATED_TO", physics[i + 1], confidence=0.9)
    kg.add_triple("curie", "DISCOVERED", "radium", confidence=0.95)
    kg.add_triple("curie", "DISCOVERED", "polonium", confidence=0.95)
    for i in range(len(music) - 1):
        kg.add_triple(music[i], "RELATED_TO", music[i + 1], confidence=0.9)
    kg.add_triple("lennon", "MEMBER_OF", "beatles", confidence=0.95)
    kg.add_triple("mccartney", "MEMBER_OF", "beatles", confidence=0.95)
    # bridge
    kg.add_triple("sorbonne", "LOCATED_NEAR", "liverpool", confidence=0.5)
    return gkg


def test_detect_communities_splits_clusters():
    gkg = _two_cluster_kg()
    communities = detect_communities(gkg.kg, min_size=3)
    assert len(communities) >= 2
    memberships = [set(c["entity_ids"]) for c in communities]
    # The two dense cores must land in different communities
    assert not any({"curie", "beatles"} <= m for m in memberships)
    assert any("curie" in m and "radium" in m for m in memberships)
    assert any("beatles" in m and "lennon" in m for m in memberships)


def test_build_summaries_with_stub_and_serialization():
    gkg = _two_cluster_kg()

    def fake_summarizer(facts):
        return {"title": "Test Cluster", "summary": f"Cluster with {len(facts.splitlines())} facts."}

    communities = build_community_summaries(gkg, summarize_fn=fake_summarizer)
    assert all(c["summary"] for c in communities)
    assert gkg.communities == communities

    restored = GovernedKnowledgeGraph.from_dict(gkg.to_dict())
    assert restored.communities == communities


def test_community_context_lexical_fallback():
    gkg = _two_cluster_kg()
    build_community_summaries(
        gkg,
        summarize_fn=lambda facts: {
            "title": "Radioactivity research" if "curie" in facts else "British rock band",
            "summary": (
                "Marie Curie discovered radium and polonium."
                if "curie" in facts
                else "The Beatles with Lennon and McCartney recorded Abbey Road."
            ),
        },
    )
    block = community_context(gkg, "Who discovered radium and polonium?", top_k=1)
    assert "Radioactivity" in block or "Curie" in block
    assert "Beatles" not in block


def test_community_context_empty_when_no_communities():
    gkg = GovernedKnowledgeGraph(governance_mode="audit_only")
    assert community_context(gkg, "anything") == ""


def test_ppr_ranks_seed_neighbourhood_highest():
    gkg = _two_cluster_kg()
    ranked = personalized_pagerank(gkg.kg, {"curie"}, top_k=5)
    assert ranked, "PPR should return results when seed exists"
    top_entities = [eid for eid, _ in ranked]
    assert top_entities[0] == "curie"
    # Physics cluster should dominate the top ranks over the music cluster
    physics_hits = sum(1 for e in top_entities if e in {"curie", "becquerel", "radium", "polonium", "sorbonne"})
    assert physics_hits >= 4


def test_ppr_empty_for_unknown_seed():
    gkg = _two_cluster_kg()
    assert personalized_pagerank(gkg.kg, {"nonexistent"}, top_k=5) == []


def test_ppr_skips_superseded_triples():
    gkg = _two_cluster_kg()
    triple = gkg.kg.find_triple("curie", "DISCOVERED", "radium")
    new = gkg.kg.add_triple("curie", "DISCOVERED", "actinium_wrong", confidence=0.9)
    gkg.kg.mark_superseded(triple, by=new)
    ranked = dict(personalized_pagerank(gkg.kg, {"curie"}, top_k=10))
    assert "radium" in ranked  # still connected via RELATED_TO chain
    communities = detect_communities(gkg.kg, min_size=3)
    assert communities  # detection still works with superseded edges present
