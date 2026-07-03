"""Coref must MERGE entities, never DELETE them (DocRED doc-3 collapse,
evaluation/DocRED/LESSONS.md Lesson 1).

The LLM often returns only multi-mention clusters and omits singletons; stage 4
must pass unclaimed entities through instead of treating the groups as the
complete entity list.
"""
import pytest

from multi_agent_kg.agents.entity_extractor import (
    EntityExtractor,
    _is_generic_reference,
)


def _extractor(llm_response):
    """EntityExtractor without full __init__ — only what stage 4 touches."""
    ex = EntityExtractor.__new__(EntityExtractor)
    ex.shared_memory = None
    if isinstance(llm_response, Exception):
        def fake(*a, **k):
            raise llm_response
    else:
        def fake(*a, **k):
            return llm_response
    ex.call_llm = fake
    return ex


def _ents(*names):
    return [{"id": n.lower().replace(" ", "_"), "text": n, "type": "THING",
             "confidence": 0.9} for n in names]


PROTAGONIST_ONLY = {
    "entity_groups": [{
        "canonical_id": "ramey_idriss",
        "canonical_name": "Ramey Idriss",
        "type": "PERSON",
        "mentions": ["Ramey Idriss", "Ramez Idriss", "Ramey"],
        "is_known_entity": False,
    }]
}


def test_singletons_pass_through_when_llm_returns_one_group():
    """The doc-3 collapse: 5 in, LLM claims only the protagonist cluster."""
    ex = _extractor(PROTAGONIST_ONLY)
    entities = _ents("Ramey Idriss", "Ramez Idriss",
                     "Los Angeles Community College",
                     "The Woody Woodpecker Song", "George Tibbles")
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    names = {e.get("text") for e in resolved}
    # protagonist merged into one node, all three singletons survive
    assert "Los Angeles Community College" in names
    assert "The Woody Woodpecker Song" in names
    assert "George Tibbles" in names
    assert "Ramey Idriss" in names
    assert len(resolved) == 4  # 2 aliases merged + 3 singletons


def test_unclaimed_pronouns_are_still_dropped():
    ex = _extractor(PROTAGONIST_ONLY)
    entities = _ents("Ramey Idriss", "he", "Los Angeles Community College")
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    names = {e.get("text") for e in resolved}
    assert "he" not in names
    assert "Los Angeles Community College" in names


def test_failed_llm_call_passes_batch_through():
    ex = _extractor(RuntimeError("vllm down"))
    entities = _ents("Alpha Corp", "Beta Labs")
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    assert {e["text"] for e in resolved} == {"Alpha Corp", "Beta Labs"}


def test_garbage_llm_response_passes_batch_through():
    ex = _extractor("chain of thought prose, not JSON")
    entities = _ents("Alpha Corp", "Beta Labs")
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    assert {e["text"] for e in resolved} == {"Alpha Corp", "Beta Labs"}


def test_claimed_entities_are_not_duplicated_by_passthrough():
    ex = _extractor(PROTAGONIST_ONLY)
    entities = _ents("Ramey Idriss", "Ramez Idriss", "Ramey")
    resolved = ex._stage4_coreference_resolution("text", entities, [])
    assert len(resolved) == 1
    assert resolved[0]["text"] == "Ramey Idriss"
    # merged group keeps the strongest member confidence
    assert resolved[0]["confidence"] == pytest.approx(0.9)


def test_generic_reference_helper():
    assert _is_generic_reference("he")
    assert _is_generic_reference("the approach")
    assert _is_generic_reference("")
    assert not _is_generic_reference("The Beatles")
    assert not _is_generic_reference("Los Angeles Community College")
