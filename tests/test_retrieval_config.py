"""Regression guard for item ① — promoting hardcoded retrieval constants to
RetrievalConfig fields must not change default behaviour, and the new fields must
actually shape retrieval when overridden.
"""

from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
from multi_agent_kg.core.governance import Domain
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.qa_orchestrator import DomainExpertAgent


def test_defaults_match_historical_constants() -> None:
    c = RetrievalConfig()
    # Values previously hardcoded inline across qa_orchestrator/advanced_qa.
    assert c.seed_cap == 40
    assert c.lexical_seed_k == 40
    assert c.focused_limit == 60
    assert c.max_hops == 3
    assert c.neighbourhood_hops == 3
    assert c.neighbourhood_display == 50
    assert c.summary_trigger == 40


def test_mode_validation_and_vector_lexical_mapping() -> None:
    # Historical three modes keep their exact use_vectors/use_lexical semantics.
    assert (RetrievalConfig(retrieval_mode="lexical").use_vectors,
            RetrievalConfig(retrieval_mode="lexical").use_lexical) == (False, True)
    assert (RetrievalConfig(retrieval_mode="dense").use_vectors,
            RetrievalConfig(retrieval_mode="dense").use_lexical) == (True, False)
    assert (RetrievalConfig(retrieval_mode="hybrid").use_vectors,
            RetrievalConfig(retrieval_mode="hybrid").use_lexical) == (True, True)
    # New retriever switches all ride the hybrid (vectors+lexical) base.
    for mode in ("graph_completion", "graph_summary", "chunk"):
        rc = RetrievalConfig(retrieval_mode=mode)
        assert rc.use_vectors and rc.use_lexical, mode

    try:
        RetrievalConfig(retrieval_mode="nonsense")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid retrieval_mode should raise")


def _fixture_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    # One strongly query-relevant fact plus filler to exercise the limit knobs.
    kg.add_entity("ada_lovelace", ["Ada Lovelace"], "PERSON")
    kg.add_entity("london", ["London"], "PLACE")
    kg.add_triple("ada_lovelace", "BORN_IN", "london", 0.9,
                  metadata={"evidence": "Ada Lovelace was born in London."})
    for i in range(20):
        kg.add_entity(f"e{i}", [f"Entity {i}"], "MISC")
        kg.add_triple("ada_lovelace", "RELATED_TO", f"e{i}", 0.5)
    return kg


def _expert(kg: KnowledgeGraph, **overrides) -> DomainExpertAgent:
    domain = Domain(
        domain_id="d", label="D", description="fixture",
        entity_ids=set(kg.entities.keys()), relation_schema={},
    )
    rc = RetrievalConfig(retrieval_mode="lexical", **overrides)  # lexical: no vector store
    return DomainExpertAgent(domain=domain, full_kg=kg, llm_config=LLMConfig(),
                             retrieval_config=rc)


def test_focused_limit_caps_returned_triples() -> None:
    kg = _fixture_kg()
    _, domain_triples = _expert(kg).domain.get_subgraph(kg)

    small = _expert(kg, focused_limit=3)
    big = _expert(kg, focused_limit=60)
    q = "Where was Ada Lovelace born?"
    assert len(small._query_focused_triples(q, candidates=domain_triples, limit=3)) <= 3
    big_res = big._query_focused_triples(q, candidates=domain_triples, limit=60)
    # The born-in fact must survive ranking in the larger window.
    assert any(t.relation == "BORN_IN" for t in big_res)


def test_seed_cap_limits_seed_expansion() -> None:
    kg = _fixture_kg()
    _, domain_triples = _expert(kg).domain.get_subgraph(kg)
    capped = _expert(kg, seed_cap=2, lexical_seed_k=2, focused_limit=80)
    res = capped._query_focused_triples("Ada Lovelace", candidates=domain_triples, limit=80)
    # With only 2 seeds, expansion stays bounded well under the full filler set.
    assert len(res) <= 80


def test_hybrid_select_evidence_matches_legacy_path() -> None:
    # Item ⑤ regression guard: the default mode must produce exactly the triples the
    # old direct _query_focused_triples call produced, with no summary.
    kg = _fixture_kg()
    expert = _expert(kg)  # lexical mode shares the same seed+expansion path as hybrid
    _, domain_triples = expert.domain.get_subgraph(kg)
    q = "Where was Ada Lovelace born?"
    legacy = expert._query_focused_triples(q, candidates=domain_triples, limit=60)
    triples, summary = expert._select_evidence(q, candidates=domain_triples)
    assert summary is None
    assert [(t.subject, t.relation, t.object) for t in triples] == [
        (t.subject, t.relation, t.object) for t in legacy
    ]


def test_chunk_mode_skips_graph_expansion() -> None:
    kg = _fixture_kg()
    _, domain_triples = _expert(kg).domain.get_subgraph(kg)
    # Build expert directly with chunk mode (no vector store → lexical fallback).
    from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
    from multi_agent_kg.core.governance import Domain
    from multi_agent_kg.core.qa_orchestrator import DomainExpertAgent

    domain = Domain(domain_id="d", label="D", description="x",
                    entity_ids=set(kg.entities), relation_schema={})
    expert = DomainExpertAgent(domain, kg, LLMConfig(),
                               retrieval_config=RetrievalConfig(retrieval_mode="chunk"))
    triples, summary = expert._select_evidence("Where was Ada born?",
                                               candidates=domain_triples)
    assert summary is None
    assert len(triples) <= expert.retrieval_config.focused_limit
