"""
Graph traversal helpers shared by QA and governance modules.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple


def find_paths(
    kg: KnowledgeGraph,
    source: str,
    target: str,
    max_hops: int = 3,
) -> List[List[Triple]]:
    """Find paths from source to target within max_hops using BFS."""
    from collections import deque

    from multi_agent_kg.core.kg_operations import normalize_entity_name

    adj: Dict[str, List[Triple]] = {}
    for triple in kg.triples:
        adj.setdefault(triple.subject, []).append(triple)
        adj.setdefault(triple.object, []).append(triple)

    source_n = normalize_entity_name(source)
    target_n = normalize_entity_name(target)

    def _match(entity_id: str, query: str) -> bool:
        normalized = normalize_entity_name(entity_id)
        return normalized == query or query in normalized or normalized in query

    source_ids = {entity_id for entity_id in adj if _match(entity_id, source_n)}
    target_ids = {entity_id for entity_id in adj if _match(entity_id, target_n)}
    if not source_ids or not target_ids:
        return []

    paths: List[List[Triple]] = []
    queue: deque = deque()
    for entity_id in source_ids:
        queue.append((entity_id, [], {entity_id}))

    while queue:
        node, path, visited = queue.popleft()
        if len(path) > max_hops:
            continue
        if node in target_ids and path:
            paths.append(path)
            continue
        for triple in adj.get(node, []):
            nxt = triple.object if triple.subject == node else triple.subject
            if nxt not in visited:
                queue.append((nxt, path + [triple], visited | {nxt}))

    return paths


def paths_to_text(paths: List[List[Triple]]) -> str:
    """Render multi-hop paths as human-readable text."""
    if not paths:
        return "No multi-hop paths found."
    lines = [f"Found {len(paths)} path(s):"]
    for index, path in enumerate(paths[:10], 1):
        hops = " → ".join(
            f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"
            for triple in path
        )
        lines.append(
            f"  Path {index} ({len(path)} hop{'s' if len(path) > 1 else ''}): {hops}"
        )
    if len(paths) > 10:
        lines.append(f"  ... and {len(paths) - 10} more paths")
    return "\n".join(lines)


def neighbourhood(kg: KnowledgeGraph, entity_id: str, hops: int = 2) -> List[Triple]:
    """Return all triples within hops of entity_id using normalized traversal."""
    from collections import deque

    from multi_agent_kg.core.kg_operations import normalize_for_matching

    adj: Dict[str, List[Tuple[Triple, str]]] = {}
    for triple in kg.triples:
        subject_norm = normalize_for_matching(triple.subject)
        object_norm = normalize_for_matching(triple.object)
        adj.setdefault(subject_norm, []).append((triple, triple.object))
        adj.setdefault(object_norm, []).append((triple, triple.subject))

    seed = normalize_for_matching(entity_id)
    visited: Set[str] = set()
    frontier: deque = deque([(seed, 0)])
    collected: List[Triple] = []
    seen_triples: Set[int] = set()

    while frontier:
        node_norm, depth = frontier.popleft()
        if node_norm in visited or depth > hops:
            continue
        visited.add(node_norm)
        for triple, other_raw in adj.get(node_norm, []):
            triple_id = id(triple)
            if triple_id not in seen_triples:
                seen_triples.add(triple_id)
                collected.append(triple)
            other_norm = normalize_for_matching(other_raw)
            if other_norm not in visited and depth + 1 <= hops:
                frontier.append((other_norm, depth + 1))
    return collected


__all__ = [
    "find_paths",
    "paths_to_text",
    "neighbourhood",
]
