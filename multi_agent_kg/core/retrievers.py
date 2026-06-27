"""Retriever abstraction (Cognee-inspired BaseRetriever pattern).

A retriever turns a query into grounded evidence in three explicit stages:

    retrieve_objects(query)        -> raw graph objects (triples)
    context_from_objects(objects)  -> (focused_triples, summary_text_or_None)
    completion_from_context(...)   -> an answer string

The experts (DomainExpertAgent, FallbackGraphExpert, ActiveExplorerExpert) only
use the first two stages: they keep their own richer synthesis (provenance,
coverage/confidence, critic) as ``completion``. The third stage here is a thin
default used for standalone/testing use, so the abstraction is complete without
forcing every expert through it.

Design rule (matches the rest of the system): retrievers carry NO benchmark
identity and add NO new behaviour in Phase 1 — each subclass simply wraps the
agent's existing, already-tested helper methods so the ``hybrid`` path stays
byte-identical. Phase 2 (graph_completion) overrides ``retrieve_objects`` to add
multi-collection vector seeding + true k-hop expansion.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Evidence is (focused_triples, summary_text_or_None) — the historical
# _select_evidence return shape, preserved exactly.
Evidence = Tuple[List[Any], Optional[str]]


class BaseRetriever:
    """Wraps a host expert and exposes the three-stage retrieval contract.

    ``agent`` must provide ``retrieval_config`` and the helper methods
    ``_query_focused_triples`` / ``_chunk_select_triples`` / ``_summarize_subgraph``
    (all defined on DomainExpertAgent, inherited by the other experts).
    """

    def __init__(self, agent: Any):
        self.agent = agent

    # -- stage 1 -----------------------------------------------------------
    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        """Default: the historical lexical∪dense seed + graph-expansion path."""
        return self.agent._query_focused_triples(
            query,
            candidates=candidates,
            limit=self.agent.retrieval_config.focused_limit,
        )

    # -- stage 2 -----------------------------------------------------------
    def context_from_objects(self, objects: List[Any], query: str) -> Evidence:
        """Default: raw triples, no summary (hybrid/lexical/dense/graph_completion)."""
        return objects, None

    # -- stages 1+2 bundled (the old _select_evidence return) --------------
    def select_evidence(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> Evidence:
        objects = self.retrieve_objects(query, candidates)
        return self.context_from_objects(objects, query)

    # -- stage 3 (thin default; experts override via their own answer()) ---
    def completion_from_context(self, context: str, query: str) -> str:
        """Minimal LLM completion over a prepared context string.

        Experts do not call this — they keep their richer synthesis. Provided so
        the abstraction is whole and usable standalone (e.g. tests, simple CLIs).
        """
        from multi_agent_kg.llm.openai_client import chat_completion_json

        prompt = (
            "Answer the QUESTION using ONLY the CONTEXT. Be concise and do not "
            "add information that is not in the context.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
            'Return JSON: {"answer": "..."}'
        )
        try:
            result = chat_completion_json(
                messages=[
                    {"role": "system", "content": "You answer strictly from the given context."},
                    {"role": "user", "content": prompt},
                ],
                model=self.agent.llm_config.model,
                temperature=0.1,
            )
            if isinstance(result, dict):
                return str(result.get("answer", "")).strip()
        except Exception:
            pass
        return ""


class HybridRetriever(BaseRetriever):
    """Default fused retrieval: lexical ∪ dense seeds + graph expansion.

    Used for ``hybrid`` / ``lexical`` / ``dense`` (the seed mix is gated inside
    ``_query_focused_triples`` by use_lexical/use_vectors) and, in Phase 1, also
    for ``graph_completion`` (Phase 2 promotes that to its own subclass).
    """


class ChunkRetriever(BaseRetriever):
    """``chunk`` mode: top-k triples by dense similarity, NO graph expansion."""

    def retrieve_objects(
        self, query: str, candidates: Optional[List[Any]] = None
    ) -> List[Any]:
        return self.agent._chunk_select_triples(query, candidates)


class GraphSummaryRetriever(HybridRetriever):
    """``graph_summary`` mode: hybrid retrieval, then LLM-summarize a large subgraph."""

    def context_from_objects(self, objects: List[Any], query: str) -> Evidence:
        summary = None
        if len(objects) > self.agent.retrieval_config.summary_trigger:
            summary = self.agent._summarize_subgraph(query, objects)
        return objects, summary


# Mode string -> retriever class. graph_completion maps to Hybrid in Phase 1.
_RETRIEVERS = {
    "hybrid": HybridRetriever,
    "lexical": HybridRetriever,
    "dense": HybridRetriever,
    "graph_completion": HybridRetriever,
    "graph_summary": GraphSummaryRetriever,
    "chunk": ChunkRetriever,
}


def get_retriever(mode: str, agent: Any) -> BaseRetriever:
    """Factory: return the retriever for ``mode`` bound to ``agent``.

    Unknown modes fall back to Hybrid (RetrievalConfig already validates the set,
    so this is defensive only).
    """
    return _RETRIEVERS.get(mode, HybridRetriever)(agent)


__all__ = [
    "BaseRetriever",
    "HybridRetriever",
    "ChunkRetriever",
    "GraphSummaryRetriever",
    "get_retriever",
]
