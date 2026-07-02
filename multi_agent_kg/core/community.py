"""
Community detection and summarization over the knowledge graph (GraphRAG-style).

Entities are clustered with Louvain over the active-triple graph; each
community of sufficient size gets an LLM-generated title + summary. The
summaries are stored on the GovernedKnowledgeGraph (serialized with it) and
indexed in the vector store, giving QA a retrievable "global" layer for
broad or thematic questions that entity-local retrieval cannot answer.
"""

from typing import Any, Callable, Dict, List, Optional

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph

COMMUNITY_SUMMARY_PROMPT = """Summarize this cluster of related facts from a knowledge graph.

FACTS:
{facts}

Write:
1. A short TITLE (max 8 words) naming what this cluster is about.
2. A SUMMARY (max 150 words) of the key entities, how they relate, and any notable attributes, dates, or values. Preserve entity names verbatim. Do not add outside knowledge.

Return JSON:
{{
    "title": "<title>",
    "summary": "<summary>"
}}"""


def detect_communities(
    kg: KnowledgeGraph,
    min_size: int = 3,
    max_communities: int = 40,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Louvain communities over the undirected active-triple graph.

    Returns unsummarized community dicts, largest first:
    {"community_id", "entity_ids", "size", "triple_count", "title", "summary"}.
    """
    import networkx as nx

    graph = nx.Graph()
    active = kg.get_active_triples()
    for triple in active:
        graph.add_edge(triple.subject, triple.object)
    if graph.number_of_nodes() < min_size:
        return []

    try:
        raw = nx.community.louvain_communities(graph, seed=seed)
    except Exception:
        return []

    communities: List[Dict[str, Any]] = []
    for members in sorted(raw, key=len, reverse=True)[:max_communities]:
        if len(members) < min_size:
            continue
        member_set = set(members)
        triple_count = sum(
            1 for t in active if t.subject in member_set and t.object in member_set
        )
        communities.append(
            {
                "community_id": f"community_{len(communities)}",
                "entity_ids": sorted(member_set),
                "size": len(member_set),
                "triple_count": triple_count,
                "title": "",
                "summary": "",
            }
        )
    return communities


def _community_facts(kg: KnowledgeGraph, community: Dict[str, Any], max_triples: int = 60) -> str:
    member_set = set(community["entity_ids"])
    internal = [
        t for t in kg.get_active_triples()
        if t.subject in member_set and t.object in member_set
    ]
    # Highest-confidence facts first so the cap keeps the strongest signal.
    internal.sort(key=lambda t: t.confidence or 0.0, reverse=True)
    return "\n".join(
        f"({t.subject}) -[{t.relation}]-> ({t.object})" for t in internal[:max_triples]
    )


def _llm_summarize(facts: str, model: Optional[str]) -> Dict[str, str]:
    from multi_agent_kg.llm.openai_client import chat_completion_json, DEFAULT_CHAT_MODEL

    result = chat_completion_json(
        [
            {
                "role": "system",
                "content": "You summarize knowledge-graph clusters faithfully and concisely.",
            },
            {"role": "user", "content": COMMUNITY_SUMMARY_PROMPT.format(facts=facts)},
        ],
        model=model or DEFAULT_CHAT_MODEL,
        temperature=0.2,
        max_tokens=2048,
    )
    if not isinstance(result, dict):
        return {}
    return {
        "title": str(result.get("title", ""))[:120],
        "summary": str(result.get("summary", ""))[:1500],
    }


def build_community_summaries(
    governed_kg: Any,
    model: Optional[str] = None,
    min_size: int = 3,
    max_communities: int = 40,
    summarize_fn: Optional[Callable[[str], Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Detect, summarize, and store communities on `governed_kg.communities`.

    `summarize_fn(facts) -> {"title", "summary"}` overrides the LLM (tests).
    A failed summary leaves title/summary empty rather than dropping the
    community — membership is still useful for retrieval.
    """
    kg = governed_kg.kg
    communities = detect_communities(kg, min_size=min_size, max_communities=max_communities)
    for community in communities:
        facts = _community_facts(kg, community)
        if not facts:
            continue
        try:
            fields = summarize_fn(facts) if summarize_fn else _llm_summarize(facts, model)
        except Exception:
            fields = {}
        community["title"] = (fields or {}).get("title", "")
        community["summary"] = (fields or {}).get("summary", "")
    governed_kg.communities = communities
    return communities


def community_text(community: Dict[str, Any]) -> str:
    """Embeddable / promptable rendering of one community."""
    members_preview = ", ".join(community.get("entity_ids", [])[:15])
    parts = [
        community.get("title", ""),
        community.get("summary", ""),
        f"Members: {members_preview}" if members_preview else "",
    ]
    return ". ".join(p for p in parts if p)


def community_context(
    governed_kg: Any,
    query: str,
    top_k: int = 3,
    vector_store: Optional[Any] = None,
) -> str:
    """Top-k community summaries relevant to `query`, as a context block.

    Uses the vector store's community index when available; otherwise falls
    back to lexical term overlap. Returns "" when nothing matches.
    """
    communities = getattr(governed_kg, "communities", None) or []
    scored_ids: List[str] = []
    index = getattr(vector_store, "community_index", None) if vector_store else None
    if index is not None and len(index) > 0:
        try:
            scored_ids = [cid for cid, _ in index.search(query, top_k=top_k)]
        except Exception:
            scored_ids = []
    if not scored_ids:
        terms = {w for w in query.lower().split() if len(w) > 2}
        lexical = []
        for community in communities:
            text = community_text(community).lower()
            overlap = sum(1 for term in terms if term in text)
            if overlap:
                lexical.append((overlap, community["community_id"]))
        scored_ids = [cid for _, cid in sorted(lexical, reverse=True)[:top_k]]
    if not scored_ids:
        return ""

    by_id = {c["community_id"]: c for c in communities}
    lines = ["Corpus-level community summaries (thematic context):"]
    for cid in scored_ids:
        community = by_id.get(cid)
        if not community or not (community.get("summary") or community.get("title")):
            continue
        title = community.get("title") or cid
        lines.append(f"  [{title}] {community.get('summary', '')}")
    return "\n".join(lines) if len(lines) > 1 else ""


__all__ = [
    "detect_communities",
    "build_community_summaries",
    "community_text",
    "community_context",
    "COMMUNITY_SUMMARY_PROMPT",
]
