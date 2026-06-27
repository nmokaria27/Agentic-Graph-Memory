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
        "graph_completion": HybridRetriever,  # promoted to its own class in Phase 2
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
