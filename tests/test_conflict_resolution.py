"""Tests for Mem0-style conflict resolution on triple admission."""

from multi_agent_kg.core import provenance as prov
from multi_agent_kg.core.conflict_resolution import (
    COEXIST,
    DISCARD_NEW,
    SUPERSEDE,
    _describe,
    normalize_resolution,
)
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import Triple, is_superseded, triple_uid


def _gkg(resolver):
    gkg = GovernedKnowledgeGraph(governance_mode="permissive", conflict_resolver=resolver)
    for eid in ("acme", "alice", "bob", "paris", "london"):
        gkg.add_entity(eid)
    return gkg


def _static_resolver(action, indices=None):
    def resolver(new_triple, existing):
        return {"action": action, "superseded_indices": indices or [], "reasoning": "test"}
    return resolver


def test_no_conflict_never_calls_resolver():
    calls = []

    def resolver(new_triple, existing):
        calls.append(new_triple)
        return {"action": COEXIST}

    gkg = _gkg(resolver)
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    gkg.propose_triple("acme", "LOCATED_IN", "paris", confidence=0.9)
    assert calls == []
    assert len(gkg.kg.get_active_triples()) == 2


def test_supersede_marks_old_and_admits_new():
    gkg = _gkg(_static_resolver(SUPERSEDE, [0]))
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    decision = gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9)

    assert decision.committed
    active = gkg.kg.get_active_triples()
    assert len(active) == 1
    assert active[0].object == "bob"

    old = [t for t in gkg.kg.triples if t.object == "alice"][0]
    assert is_superseded(old)
    assert old.metadata["superseded_by"] == triple_uid(active[0])
    assert gkg.get_stats()["conflict_resolution"]["supersede"] == 1
    assert gkg.get_stats()["conflict_resolution"]["triples_superseded"] == 1


def test_discard_new_rejects_proposal():
    gkg = _gkg(_static_resolver(DISCARD_NEW))
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    decision = gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9)

    assert decision.action == "reject"
    assert not decision.committed
    active = gkg.kg.get_active_triples()
    assert len(active) == 1
    assert active[0].object == "alice"


def test_coexist_keeps_both():
    gkg = _gkg(_static_resolver(COEXIST))
    gkg.propose_triple("alice", "MEMBER_OF", "acme", confidence=0.9)
    gkg.propose_triple("alice", "MEMBER_OF", "paris", confidence=0.9)
    assert len(gkg.kg.get_active_triples()) == 2


def test_resolver_exception_degrades_to_coexist():
    def broken(new_triple, existing):
        raise RuntimeError("resolver down")

    gkg = _gkg(broken)
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    decision = gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9)
    assert decision.committed
    assert len(gkg.kg.get_active_triples()) == 2


def test_superseded_triples_do_not_reconflict():
    gkg = _gkg(_static_resolver(SUPERSEDE, [0]))
    gkg.propose_triple("acme", "HQ_IN", "paris", confidence=0.9)
    gkg.propose_triple("acme", "HQ_IN", "london", confidence=0.9)
    # Third statement should only conflict with the active one (london)
    decision = gkg.propose_triple("acme", "HQ_IN", "paris", confidence=0.9)
    assert decision.committed
    conflicting = decision.triple.metadata["conflict_resolution"]["conflicting"]
    assert conflicting == ["acme|HQ_IN|london"]


def test_no_resolver_preserves_legacy_coexist_behavior():
    gkg = _gkg(None)
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9)
    assert len(gkg.kg.get_active_triples()) == 2


def test_normalize_resolution_handles_garbage():
    assert normalize_resolution(None, 2)["action"] == COEXIST
    assert normalize_resolution({"action": "explode"}, 2)["action"] == COEXIST
    r = normalize_resolution({"action": "supersede", "superseded_indices": [5, -1]}, 2)
    # Invalid indices on supersede -> replace all conflicting facts
    assert r["superseded_indices"] == [0, 1]
    r2 = normalize_resolution({"action": "supersede", "superseded_indices": [1]}, 2)
    assert r2["superseded_indices"] == [1]


def test_supersede_survives_serialization():
    gkg = _gkg(_static_resolver(SUPERSEDE, [0]))
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9)
    gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9)

    restored = GovernedKnowledgeGraph.from_dict(gkg.to_dict())
    active = restored.kg.get_active_triples()
    assert len(active) == 1
    assert active[0].object == "bob"


def test_describe_surfaces_document_date_when_present():
    """GB-2: a triple whose provenance carries a document_date shows date= in
    the resolver's description, giving the LLM a recency signal to act on."""
    triple = Triple(
        subject="rachel", relation="located_in", object="chicago",
        confidence=0.9, source="doc_1_abc",
        metadata={
            "provenance": prov.build_provenance(
                refs=[prov.source_ref("doc_1_abc", document_date="2022-03-01")],
                extractor="RelationExtractor",
                confidence=0.9,
            )
        },
    )
    assert "date=2022-03-01" in _describe(triple)


def test_describe_omits_date_when_absent():
    """Regression guard: undated sources (DocRED-style) produce byte-identical
    output to before document_date existed — no 'date=' segment at all."""
    triple = Triple(
        subject="rachel", relation="located_in", object="chicago",
        confidence=0.9, source="doc_1_abc",
        metadata={
            "provenance": prov.build_provenance(
                refs=[prov.source_ref("doc_1_abc")],
                extractor="RelationExtractor",
                confidence=0.9,
            )
        },
    )
    assert "date=" not in _describe(triple)


def test_describe_omits_date_with_no_provenance_at_all():
    triple = Triple(subject="rachel", relation="located_in", object="chicago", confidence=0.9)
    assert "date=" not in _describe(triple)


def test_conflict_resolver_receives_dated_descriptions():
    """Integration-style: propose_triple -> _apply_conflict_resolution wires real
    Triple objects (with provenance) into the resolver; a resolver that inspects
    _describe() output sees both triples' dates, proving the plumbing from
    GovernedKnowledgeGraph.propose_triple through to the resolver call is intact."""
    seen_descriptions = []

    def spy_resolver(new_triple, existing):
        seen_descriptions.append(_describe(new_triple))
        for t in existing:
            seen_descriptions.append(_describe(t))
        return {"action": SUPERSEDE, "superseded_indices": [0], "reasoning": "newer date"}

    gkg = _gkg(spy_resolver)
    old_prov = prov.build_provenance(
        refs=[prov.source_ref("doc_0", document_date="2022-01-01")],
        extractor="RelationExtractor", confidence=0.9,
    )
    gkg.propose_triple("acme", "CEO_IS", "alice", confidence=0.9,
                        metadata={"provenance": old_prov})
    new_prov = prov.build_provenance(
        refs=[prov.source_ref("doc_5", document_date="2023-12-25")],
        extractor="RelationExtractor", confidence=0.9,
    )
    decision = gkg.propose_triple("acme", "CEO_IS", "bob", confidence=0.9,
                                   metadata={"provenance": new_prov})

    assert decision.committed
    assert any("date=2022-01-01" in d for d in seen_descriptions)
    assert any("date=2023-12-25" in d for d in seen_descriptions)
    old = [t for t in gkg.kg.triples if t.object == "alice"][0]
    assert is_superseded(old)


def _fake_embed(texts):
    """Deterministic trigram-hash embedding (same scheme as test_vector_index.py)."""
    import numpy as np
    dim = 64
    out = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        t = text.lower()
        for i in range(len(t) - 2):
            vec[hash(t[i:i + 3]) % dim] += 1.0
        if not vec.any():
            vec[0] = 1.0
        out.append(vec.tolist())
    return out


def test_relation_aware_conflicts_matches_embedding_similar_relation():
    """GB-2c: same-subject facts under different-but-synonymous relation names
    must surface as conflict candidates, not silently coexist forever."""
    from multi_agent_kg.core.vector_index import KGVectorStore

    gkg = GovernedKnowledgeGraph(governance_mode="permissive", conflict_resolver=None)
    gkg.vector_store = KGVectorStore(model="fake-model", embed_fn=_fake_embed)
    gkg.add_entity("personal_best_time")
    gkg.add_entity("twenty_seven_twelve")
    gkg.add_entity("twenty_five_fifty")
    gkg.add_entity("chicago")

    gkg.kg.add_triple("personal_best_time", "RACE_COMPLETION_TIME_DURATION",
                      "twenty_seven_twelve", confidence=0.9)
    unrelated = gkg.kg.add_triple("personal_best_time", "LOCATED_IN", "chicago", confidence=0.9)

    candidate = Triple(subject="personal_best_time",
                       relation="RACE_COMPLETION_TIME_DURATION_VALUE",
                       object="twenty_five_fifty", confidence=0.9)

    hits = gkg._relation_aware_conflicts(candidate)
    assert len(hits) == 1
    assert hits[0].relation == "RACE_COMPLETION_TIME_DURATION"
    assert unrelated not in hits


def test_relation_aware_conflicts_inert_without_vector_store():
    """Regression guard: no vector store attached -> zero behavior change
    (DocRED and any unvectored corpus is unaffected — control must not move)."""
    gkg = GovernedKnowledgeGraph(governance_mode="permissive", conflict_resolver=None)
    gkg.add_entity("personal_best_time")
    gkg.add_entity("twenty_seven_twelve")
    gkg.kg.add_triple("personal_best_time", "RACE_COMPLETION_TIME_DURATION",
                      "twenty_seven_twelve", confidence=0.9)

    candidate = Triple(subject="personal_best_time",
                       relation="RACE_COMPLETION_TIME_DURATION_VALUE",
                       object="twenty_five_fifty", confidence=0.9)
    assert gkg._relation_aware_conflicts(candidate) == []


def test_relation_aware_conflict_reaches_resolver_and_supersedes():
    """End-to-end: propose_triple must route a relation-variant conflict
    through the resolver, and a supersede decision must mark the old (exact-
    relation-miss) fact superseded."""
    from multi_agent_kg.core.vector_index import KGVectorStore

    def resolver(new_triple, existing):
        return {"action": SUPERSEDE, "superseded_indices": [0], "reasoning": "newer value"}

    gkg = GovernedKnowledgeGraph(governance_mode="permissive", conflict_resolver=resolver)
    gkg.vector_store = KGVectorStore(model="fake-model", embed_fn=_fake_embed)
    gkg.add_entity("personal_best_time")
    gkg.add_entity("twenty_seven_twelve")
    gkg.add_entity("twenty_five_fifty")

    gkg.propose_triple("personal_best_time", "RACE_COMPLETION_TIME_DURATION",
                       "twenty_seven_twelve", confidence=0.9)
    decision = gkg.propose_triple("personal_best_time", "RACE_COMPLETION_TIME_DURATION_VALUE",
                                  "twenty_five_fifty", confidence=0.9)

    assert decision.committed
    old = [t for t in gkg.kg.triples if t.object == "twenty_seven_twelve"][0]
    assert is_superseded(old)
