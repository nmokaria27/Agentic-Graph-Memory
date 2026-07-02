"""
Conflict resolution for triple admission (Mem0-style memory operations).

When a proposed triple conflicts with existing knowledge (same subject and
relation, different object), a resolver decides between:

- "coexist":      both statements are valid (multi-valued relation, different
                  time periods, complementary granularity) — ADD.
- "supersede":    the new statement replaces one or more existing ones
                  (single-valued relation, newer information) — UPDATE. The
                  old triples are marked superseded, never deleted.
- "discard_new":  the existing statement is correct and the new one is stale,
                  wrong, or redundant — NOOP/reject.

The GovernedKnowledgeGraph stays LLM-free: it accepts any callable
`(new_triple, existing_triples) -> dict`. This module provides the LLM-backed
implementation wired in by the orchestrator. Resolution is domain-agnostic —
cardinality is judged from the statements and their evidence, not from a
hardcoded relation list.
"""

from typing import Any, Dict, List, Optional

from multi_agent_kg.core.knowledge_graph import Triple

COEXIST = "coexist"
SUPERSEDE = "supersede"
DISCARD_NEW = "discard_new"

_VALID_ACTIONS = {COEXIST, SUPERSEDE, DISCARD_NEW}


CONFLICT_RESOLUTION_PROMPT = """You maintain a knowledge graph. A newly extracted fact conflicts with existing facts (same subject and relation, different object). Decide how to resolve it.

NEW FACT:
{new_fact}

EXISTING FACTS:
{existing_facts}

Decide ONE action:
- "coexist": both can be true simultaneously. Use when the relation naturally holds multiple values (memberships, works, children, awards), or the statements cover different time periods or different granularity (e.g., city vs country).
- "supersede": the new fact REPLACES existing fact(s). Use when the relation holds a single current value (a capital, a CEO, a spouse, a location of residence, a score) and the new fact is the more recent or better-evidenced statement.
- "discard_new": the existing fact(s) stand and the new one should be dropped. Use when the new fact is staler, contradicted by stronger evidence, or a corrupted variant of an existing fact.

Judge cardinality from the meaning of the relation and the evidence — do not assume. If evidence indicates when each statement was true, prefer temporal ordering. If genuinely uncertain, choose "coexist" (keeping both is safer than losing information).

Return JSON:
{{
    "action": "coexist" | "supersede" | "discard_new",
    "superseded_indices": [<0-based indices of EXISTING FACTS replaced; only for "supersede">],
    "reasoning": "<one sentence>"
}}"""


def _evidence_of(triple: Triple) -> str:
    """Best-effort evidence snippet from triple metadata."""
    md = triple.metadata or {}
    prov = md.get("provenance") or {}
    sentences = prov.get("evidence_sentences") or []
    if sentences:
        return str(sentences[0])[:300]
    for key in ("supporting_evidence", "evidence"):
        if md.get(key):
            return str(md[key])[:300]
    return ""


def _describe(triple: Triple) -> str:
    parts = [f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"]
    if triple.confidence is not None:
        parts.append(f"confidence={triple.confidence:.2f}")
    if triple.source:
        parts.append(f"source={triple.source}")
    evidence = _evidence_of(triple)
    if evidence:
        parts.append(f'evidence="{evidence}"')
    return " | ".join(parts)


def normalize_resolution(raw: Any, n_existing: int) -> Dict[str, Any]:
    """Coerce an arbitrary resolver response into a safe resolution dict.

    Anything malformed degrades to coexist — the pre-feature behavior.
    """
    if not isinstance(raw, dict):
        return {"action": COEXIST, "superseded_indices": [], "reasoning": "unparseable resolver output"}
    action = str(raw.get("action", "")).strip().lower()
    if action not in _VALID_ACTIONS:
        return {"action": COEXIST, "superseded_indices": [], "reasoning": "unknown action from resolver"}
    indices: List[int] = []
    if action == SUPERSEDE:
        for idx in raw.get("superseded_indices") or []:
            if isinstance(idx, int) and 0 <= idx < n_existing:
                indices.append(idx)
        if not indices:
            # Supersede with no valid target: replace all conflicting facts,
            # which matches the single-valued-relation intent.
            indices = list(range(n_existing))
    return {
        "action": action,
        "superseded_indices": indices,
        "reasoning": str(raw.get("reasoning", ""))[:500],
    }


class LLMConflictResolver:
    """LLM-backed conflict resolver, pluggable into GovernedKnowledgeGraph."""

    def __init__(self, model: Optional[str] = None, max_existing: int = 6):
        self.model = model
        self.max_existing = max_existing
        self.stats = {"calls": 0, "errors": 0}

    def __call__(self, new_triple: Triple, existing: List[Triple]) -> Dict[str, Any]:
        from multi_agent_kg.llm.openai_client import chat_completion_json, DEFAULT_CHAT_MODEL

        existing = existing[: self.max_existing]
        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            new_fact=_describe(new_triple),
            existing_facts="\n".join(
                f"[{i}] {_describe(t)}" for i, t in enumerate(existing)
            ),
        )
        self.stats["calls"] += 1
        try:
            raw = chat_completion_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful knowledge-base editor. "
                            "You resolve conflicts between facts using evidence and relation semantics."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model or DEFAULT_CHAT_MODEL,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception:
            self.stats["errors"] += 1
            return {"action": COEXIST, "superseded_indices": [], "reasoning": "resolver LLM call failed"}
        return normalize_resolution(raw, len(existing))


__all__ = [
    "COEXIST",
    "SUPERSEDE",
    "DISCARD_NEW",
    "LLMConflictResolver",
    "normalize_resolution",
    "CONFLICT_RESOLUTION_PROMPT",
]
