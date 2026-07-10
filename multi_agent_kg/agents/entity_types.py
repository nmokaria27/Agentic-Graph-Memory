"""Shared entity merge-guard helpers (GB-8 / GB-11 / GB-12).

Kept outside EntityExtractor / KnowledgeOrganizer so both can import without
circular dependencies. Blank types are not usable signals for type-compatibility
gates: UNKNOWN==UNKNOWN must not look like a positive same-type match. Distinct
numeric/date literals are never the same entity in any domain.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

_BLANK_ENTITY_TYPES = frozenset({"", "UNKNOWN", "?"})

_DIGIT_RUN = re.compile(r"\d+")
# "1,000"-style thousands separators between digit groups — collapse so
# "1,000" and "1000" share a signature.
_THOUSANDS_SEP = re.compile(r"(?<=\d)[,.](?=\d{3}(?:\D|$))")


def numeric_signature(text: Optional[str]) -> Tuple[str, ...]:
    """Digit runs of a surface form, thousands separators collapsed."""
    cleaned = _THOUSANDS_SEP.sub("", text or "")
    return tuple(_DIGIT_RUN.findall(cleaned))


def literal_conflict(a: Optional[str], b: Optional[str]) -> bool:
    """True when both surfaces carry digits and those digits differ.

    GB-12: "1911" and "1939" (or "1 million" and "100 million") denote
    different literals — no coreference can make them the same entity, so
    stage-9 dedup must never merge them regardless of type or string
    similarity. Digit-free surfaces are never blocked here.
    """
    sig_a, sig_b = numeric_signature(a), numeric_signature(b)
    if not sig_a or not sig_b:
        return False
    return sig_a != sig_b


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
