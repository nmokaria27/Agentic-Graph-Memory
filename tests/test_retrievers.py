"""Phase 1 (BaseRetriever spine) — the factory must dispatch each retrieval_mode to
the right retriever, and the wrappers must reproduce the legacy helper output so the
default ``hybrid`` path stays byte-identical after the refactor.
"""

from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
from multi_agent_kg.core.governance import Domain
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.qa_orchestrator import DomainExpertAgent
from multi_agent_kg.core.retrievers import (
    ChunkRetriever,
    GraphCompletionRetriever,
    GraphSummaryRetriever,
    HybridRetriever,
    get_retriever,
)


def _fixture_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity("ada_lovelace", ["Ada Lovelace"], "PERSON")
    kg.add_entity("london", ["London"], "PLACE")
    kg.add_triple("ada_lovelace", "BORN_IN", "london", 0.9,
                  metadata={"evidence": "Ada Lovelace was born in London."})
    for i in range(20):
        kg.add_entity(f"e{i}", [f"Entity {i}"], "MISC")
        kg.add_triple("ada_lovelace", "RELATED_TO", f"e{i}", 0.5)
    return kg


def _expert(mode: str) -> DomainExpertAgent:
    kg = _fixture_kg()
    domain = Domain(domain_id="d", label="D", description="fixture",
                    entity_ids=set(kg.entities), relation_schema={})
    rc = RetrievalConfig(retrieval_mode=mode)  # no vector store -> lexical fallback
    return DomainExpertAgent(domain=domain, full_kg=kg, llm_config=LLMConfig(),
                             retrieval_config=rc)


def test_factory_dispatches_each_mode() -> None:
    cases = {
        "hybrid": HybridRetriever,
        "lexical": HybridRetriever,
        "dense": HybridRetriever,
        "graph_completion": GraphCompletionRetriever,
        "graph_summary": GraphSummaryRetriever,
        "chunk": ChunkRetriever,
    }
    for mode, cls in cases.items():
        agent = _expert(mode)
        assert type(get_retriever(mode, agent)) is cls, mode


def test_hybrid_retriever_matches_legacy_helper() -> None:
    agent = _expert("hybrid")
    _, domain_triples = agent.domain.get_subgraph(agent.full_kg)
    q = "Where was Ada Lovelace born?"
    legacy = agent._query_focused_triples(q, candidates=domain_triples,
                                          limit=agent.retrieval_config.focused_limit)
    triples, summary = get_retriever("hybrid", agent).select_evidence(q, domain_triples)
    assert summary is None
    assert [(t.subject, t.relation, t.object) for t in triples] == [
        (t.subject, t.relation, t.object) for t in legacy
    ]


def test_chunk_retriever_skips_expansion() -> None:
    agent = _expert("chunk")
    _, domain_triples = agent.domain.get_subgraph(agent.full_kg)
    triples, summary = get_retriever("chunk", agent).select_evidence(
        "Where was Ada born?", domain_triples)
    assert summary is None
    assert len(triples) <= agent.retrieval_config.focused_limit


def test_graph_summary_only_summarizes_above_trigger() -> None:
    # Below trigger -> no summary (no LLM call); the trigger gate is the contract.
    agent = _expert("graph_summary")
    agent.retrieval_config.summary_trigger = 10_000  # force below-threshold
    _, domain_triples = agent.domain.get_subgraph(agent.full_kg)
    triples, summary = get_retriever("graph_summary", agent).select_evidence(
        "Ada", domain_triples)
    assert summary is None


def test_graph_completion_falls_back_without_vector_store() -> None:
    # No vector store -> _vectors_on False -> inherited hybrid path, never crashes.
    agent = _expert("graph_completion")
    _, domain_triples = agent.domain.get_subgraph(agent.full_kg)
    triples, summary = get_retriever("graph_completion", agent).select_evidence(
        "Where was Ada Lovelace born?", domain_triples)
    assert summary is None
    assert triples  # produced evidence via fallback


# --- vector-backed graph_completion (Phase 2 core behaviour) ----------------

_DIM = 64


def _fake_embed_one(text: str):
    import numpy as np
    vec = np.zeros(_DIM, dtype=np.float32)
    text = text.lower()
    for i in range(len(text) - 2):
        vec[hash(text[i:i + 3]) % _DIM] += 1.0
    if not vec.any():
        vec[0] = 1.0
    return vec.tolist()


def _fake_embed(texts):
    return [_fake_embed_one(t) for t in texts]


def _vector_expert(mode: str):
    from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
    from multi_agent_kg.core.vector_index import KGVectorStore

    kg = KnowledgeGraph()
    kg.add_entity("ada_lovelace", ["Ada Lovelace"], "PERSON")
    kg.add_entity("london", ["London"], "PLACE")
    kg.add_entity("analytical_engine", ["Analytical Engine"], "MACHINE")
    gkg = GovernedKnowledgeGraph(kg=kg, governance_mode="audit_only")
    gkg.propose_triple("ada_lovelace", "BORN_IN", "london", confidence=0.9,
                       metadata={"evidence": "Ada Lovelace was born in London."})
    gkg.propose_triple("ada_lovelace", "WORKED_ON", "analytical_engine", confidence=0.9)
    for i in range(15):
        kg.add_entity(f"e{i}", [f"Filler {i}"], "MISC")
        gkg.propose_triple("london", "NEAR", f"e{i}", confidence=0.4)

    store = KGVectorStore(model="fake-model", embed_fn=_fake_embed)
    store.build(gkg)

    domain = Domain(domain_id="d", label="D", description="fixture",
                    entity_ids=set(kg.entities), relation_schema={})
    rc = RetrievalConfig(retrieval_mode=mode, entity_min_score=0.0)
    return DomainExpertAgent(domain=domain, full_kg=kg, llm_config=LLMConfig(),
                             vector_store=store, retrieval_config=rc), kg


def test_graph_completion_vector_seeded_khop_finds_answer() -> None:
    agent, kg = _vector_expert("graph_completion")
    _, domain_triples = agent.domain.get_subgraph(kg)
    triples, summary = get_retriever("graph_completion", agent).select_evidence(
        "Where was Ada Lovelace born?", domain_triples)
    assert summary is None
    rels = {(t.subject, t.relation, t.object) for t in triples}
    # The seed fact must be present, and k-hop expansion must pull in the
    # 1-hop neighbour reachable through the shared 'ada_lovelace' node.
    assert ("ada_lovelace", "BORN_IN", "london") in rels
    assert ("ada_lovelace", "WORKED_ON", "analytical_engine") in rels


def test_normalize_scores_minmax() -> None:
    from multi_agent_kg.core.vector_index import normalize_scores
    assert normalize_scores([]) == []
    out = dict(normalize_scores([("a", 0.2), ("b", 0.7), ("c", 0.45)]))
    assert out["a"] == 0.0 and out["b"] == 1.0
    assert 0.0 < out["c"] < 1.0
    # Degenerate (all equal) -> every item maps to 1.0, no div-by-zero.
    assert dict(normalize_scores([("a", 0.5), ("b", 0.5)])) == {"a": 1.0, "b": 1.0}
