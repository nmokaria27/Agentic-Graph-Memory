"""Tests for the vector index layer using deterministic fake embeddings."""

import math

import numpy as np
import pytest

from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.vector_index import (
    KGVectorStore,
    VectorIndex,
    rrf_fuse,
    triple_key,
    verbalize_triple,
)


# ---------------------------------------------------------------------------
# Fake embeddings: bag-of-character-trigrams hashed into a fixed-size vector.
# Deterministic, no server, and similar strings get similar vectors.
# ---------------------------------------------------------------------------

_DIM = 64


def _fake_embed_one(text: str):
    vec = np.zeros(_DIM, dtype=np.float32)
    text = text.lower()
    for i in range(len(text) - 2):
        gram = text[i : i + 3]
        vec[hash(gram) % _DIM] += 1.0
    if not vec.any():
        vec[0] = 1.0
    return vec.tolist()


def fake_embed(texts):
    return [_fake_embed_one(t) for t in texts]


def make_index():
    return VectorIndex(model="fake-model", embed_fn=fake_embed)


def test_upsert_and_search():
    index = make_index()
    index.upsert({
        "apple_inc": "Apple Inc. technology company founded by Steve Jobs",
        "steve_jobs": "Steve Jobs co-founder of Apple",
        "earthquake": "2010 Haiti earthquake natural disaster",
    })
    assert len(index) == 3
    results = index.search("Apple Inc. company", top_k=2)
    assert results[0][0] == "apple_inc"
    assert results[0][1] > results[1][1]


def test_upsert_skips_unchanged_hashes():
    index = make_index()
    assert index.upsert({"a": "hello world text"}) == 1
    assert index.upsert({"a": "hello world text"}) == 0
    assert index.upsert({"a": "different text now"}) == 1
    assert len(index) == 1


def test_upsert_updates_vector_in_place():
    index = make_index()
    index.upsert({"a": "apple fruit orchard", "b": "earthquake disaster zone"})
    index.upsert({"a": "earthquake seismic event"})
    results = index.search("earthquake", top_k=2)
    assert {r[0] for r in results} == {"a", "b"}


def test_remove():
    index = make_index()
    index.upsert({"a": "apple text", "b": "banana text", "c": "cherry text"})
    index.remove(["b"])
    assert len(index) == 2
    assert "b" not in index
    assert {r[0] for r in index.search("text", top_k=5)} == {"a", "c"}


def test_min_score_filters():
    index = make_index()
    index.upsert({"a": "completely unrelated zebra xylophone"})
    assert index.search("apple company stock", top_k=5, min_score=0.99) == []


def test_save_load_roundtrip(tmp_path):
    index = make_index()
    index.upsert({"a": "apple text", "b": "banana text"})
    path = str(tmp_path / "index.npz")
    index.save(path)

    loaded = VectorIndex.load(path, model="fake-model", embed_fn=fake_embed)
    assert loaded is not None
    assert loaded.ids == index.ids
    assert loaded.search("apple", top_k=1)[0][0] == index.search("apple", top_k=1)[0][0]

    # Model mismatch → discard
    assert VectorIndex.load(path, model="other-model") is None


def test_empty_index_search():
    index = make_index()
    assert index.search("anything", top_k=5) == []


def _build_governed_kg():
    kg = KnowledgeGraph()
    kg.add_entity("apple_inc", labels=["Apple Inc.", "Apple"], entity_type="ORG")
    kg.add_entity("steve_jobs", labels=["Steve Jobs"], entity_type="PERSON")
    kg.add_entity("cupertino", labels=["Cupertino"], entity_type="CITY")
    gkg = GovernedKnowledgeGraph(kg=kg, governance_mode="audit_only")
    gkg.propose_triple("steve_jobs", "FOUNDED", "apple_inc", confidence=0.9)
    gkg.propose_triple("apple_inc", "HEADQUARTERED_IN", "cupertino", confidence=0.9)
    return gkg


def test_kg_vector_store_build_and_search():
    gkg = _build_governed_kg()
    store = KGVectorStore(model="fake-model", embed_fn=fake_embed)
    store.build(gkg)

    assert len(store.entity_index) == 3
    assert len(store.triple_index) == 2

    hits = store.entity_index.search("Steve Jobs founder", top_k=1)
    assert hits[0][0] == "steve_jobs"

    triple_hits = store.triple_index.search("who founded apple", top_k=1)
    assert "FOUNDED" in triple_hits[0][0]


def test_kg_vector_store_dirty_refresh():
    gkg = _build_governed_kg()
    store = KGVectorStore(model="fake-model", embed_fn=fake_embed)
    store.build(gkg)

    decision = gkg.propose_triple("steve_jobs", "BORN_IN", "cupertino", confidence=0.9)
    assert decision.committed
    new_triple = decision.triple
    store.mark_dirty(triples=[new_triple])
    count = store.refresh()
    # new triple + both endpoint entities re-embedded
    assert count >= 1
    assert triple_key(new_triple) in store.triple_index

    # second refresh is a no-op
    assert store.refresh() == 0


def test_kg_vector_store_save_load_dir(tmp_path):
    gkg = _build_governed_kg()
    store = KGVectorStore(model="fake-model", embed_fn=fake_embed)
    store.build(gkg)
    directory = str(tmp_path / "vectors")
    store.save_dir(directory)

    loaded = KGVectorStore.load_dir(directory, gkg, model="fake-model", embed_fn=fake_embed)
    assert loaded is not None
    assert len(loaded.entity_index) == 3

    # Mutate KG → hash mismatch → load refuses (forces rebuild)
    gkg.propose_triple("steve_jobs", "LIVED_IN", "cupertino", confidence=0.9)
    assert KGVectorStore.load_dir(directory, gkg, model="fake-model", embed_fn=fake_embed) is None


def test_verbalize_triple_includes_evidence():
    gkg = _build_governed_kg()
    decision = gkg.propose_triple(
        "apple_inc", "MAKES", "iphone", confidence=0.9,
        metadata={"evidence": "Apple makes the iPhone."},
    )
    text = verbalize_triple(decision.triple)
    assert "apple inc makes iphone" in text
    assert "Apple makes the iPhone." in text


def test_rrf_fuse_exact_match_wins():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])
    assert fused[0] == "b"  # appears high in both lists


def test_rrf_fuse_weights():
    fused = rrf_fuse([["a"], ["b"]], weights=[2.0, 1.0])
    assert fused[0] == "a"


# ---------------------------------------------------------------------------
# Hybrid QA entity linking (A5)
# ---------------------------------------------------------------------------


def test_qa_orchestrator_hybrid_entity_linking():
    from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
    from multi_agent_kg.core.qa_orchestrator import QAOrchestrator

    gkg = _build_governed_kg()
    store = KGVectorStore(model="fake-model", embed_fn=fake_embed)
    store.build(gkg)

    config = RetrievalConfig(retrieval_mode="hybrid", entity_min_score=0.0)
    orchestrator = QAOrchestrator(
        governed_kg=gkg,
        llm_config=LLMConfig(model="test-model"),
        vector_store=store,
        retrieval_config=config,
    )

    # Exact lexical mention: must still resolve, and rank first via RRF weight.
    linked = orchestrator._extract_query_entities("Where is Apple Inc. headquartered?")
    assert "apple_inc" in linked
    assert linked[0] == "apple_inc"

    # Paraphrase with no exact entity mention: lexical finds nothing,
    # dense path must still return candidates.
    linked = orchestrator._extract_query_entities("town where the corporation has offices")
    assert linked  # dense candidates present

    # Lexical-only ablation mode ignores the vector index.
    lexical_orch = QAOrchestrator(
        governed_kg=gkg,
        llm_config=LLMConfig(model="test-model"),
        vector_store=store,
        retrieval_config=RetrievalConfig(retrieval_mode="lexical"),
    )
    assert lexical_orch._extract_query_entities("town where the corporation has offices") == []


# ---------------------------------------------------------------------------
# Embedding resolution tier in the knowledge organizer (A6/B2)
# ---------------------------------------------------------------------------


def _make_organizer(monkeypatch):
    from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
    from multi_agent_kg.core.config import LLMConfig

    monkeypatch.setattr(
        "multi_agent_kg.core.vector_index.embed_passages",
        lambda texts, model=None: fake_embed(texts),
    )
    monkeypatch.setattr(
        "multi_agent_kg.core.vector_index.embed_query",
        lambda text, model=None: _fake_embed_one(text),
    )
    gkg = GovernedKnowledgeGraph(kg=KnowledgeGraph(), governance_mode="audit_only")
    organizer = KnowledgeOrganizer(
        knowledge_graph=gkg.kg,
        governed_kg=gkg,
        llm_config=LLMConfig(model="test-model"),
    )
    organizer._embeddings_available = True
    return organizer, gkg


def test_resolution_embedding_tier_avoids_unresolved_duplicate(monkeypatch):
    organizer, gkg = _make_organizer(monkeypatch)
    entities = [
        {"id": "international_business_machines", "text": "International Business Machines",
         "type": "ORG", "mentions": [], "confidence": 0.9},
        {"id": "cloud_computing", "text": "cloud computing",
         "type": "TECHNOLOGY", "mentions": [], "confidence": 0.9},
    ]
    triples = [
        # Surface form differs from catalog text: singular "Machine".
        {"subject": "International Business Machine", "relation": "DEVELOPS",
         "object": "cloud computing", "confidence": 0.9},
    ]
    organizer._integrate_to_kg(entities, triples, document_id="doc1")

    unresolved = [e for e in gkg.entities.values() if e.type == "UNRESOLVED"]
    assert unresolved == []
    assert len(gkg.triples) == 1
    assert gkg.triples[0].subject == "international_business_machines"
    assert organizer.integration_stats["embedding_resolutions"]


def test_resolution_skips_unresolved_below_threshold(monkeypatch):
    organizer, gkg = _make_organizer(monkeypatch)
    entities = [
        {"id": "apple_inc", "text": "Apple Inc.", "type": "ORG",
         "mentions": [], "confidence": 0.9},
    ]
    triples = [
        {"subject": "quantum flux capacitor", "relation": "RELATED_TO",
         "object": "Apple Inc.", "confidence": 0.9},
    ]
    organizer._integrate_to_kg(entities, triples, document_id="doc1")

    unresolved = [e for e in gkg.entities.values() if e.type == "UNRESOLVED"]
    assert unresolved == []
    assert len(gkg.triples) == 0
    assert organizer.integration_stats["resolve_misses"]
    assert organizer.integration_stats["skipped_triple_reasons"]["unresolved_entity"] == 1
