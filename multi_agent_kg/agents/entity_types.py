"""Shared entity-type helpers (GB-8 / GB-11).

Kept outside EntityExtractor / KnowledgeOrganizer so both can import without
circular dependencies. Blank types are not usable signals for type-compatibility
gates: UNKNOWN==UNKNOWN must not look like a positive same-type match.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

_BLANK_ENTITY_TYPES = frozenset({"", "UNKNOWN", "?"})


def is_blank_entity_type(etype: Optional[str]) -> bool:
    return (etype or "").strip().upper() in _BLANK_ENTITY_TYPES


def inherit_type_from_members(
    llm_type: Optional[str],
    member_types: List[str],
) -> str:
    """Prefer a real LLM group type; else majority-vote member extraction types.

    GB-11: coref previously defaulted missing LLM types to ``UNKNOWN``, which
    erased the extractor's PERSON/LOCATION/… signal before stage 9 and left
    GB-8's type-compatibility gate inert (every entity looked same-typed).
    """
    if llm_type and not is_blank_entity_type(llm_type):
        return llm_type.strip()
    real = [t.strip() for t in member_types if t and not is_blank_entity_type(t)]
    if not real:
        return (llm_type or "UNKNOWN").strip() or "UNKNOWN"
    winner = Counter(t.upper() for t in real).most_common(1)[0][0]
    for t in real:
        if t.upper() == winner:
            return t
    return real[0]
