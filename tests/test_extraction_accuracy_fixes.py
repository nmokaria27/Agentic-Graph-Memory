"""Regression tests for two accuracy fixes:

1. Co-occurrence filter must use the closest pair of occurrences, not the
   first occurrence of each endpoint (a repeated entity whose first mention is
   far from the object must not be wrongly dropped).
2. Coreference resolution must propagate extraction confidence instead of
   flattening every group to 0.7/0.8, and must merge cross-batch duplicate
   groups that share a canonical id.
"""

from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent
from multi_agent_kg.core import LLMConfig


def _verifier() -> ExtractionVerificationAgent:
    return ExtractionVerificationAgent(llm_config=LLMConfig(model="test-model"))


def test_cooccurrence_filter_keeps_triple_via_closest_mention():
    # "alpha" appears far away first, then right next to "beta". The closest
    # pair is adjacent, so the triple must survive even with a tiny window.
    filler = "x " * 2000  # ~4000 chars, far beyond char_window=12 for window=3
    text = f"alpha {filler} alpha beta"
    triples = [{"subject": "alpha", "relation": "REL", "object": "beta"}]

    kept = _verifier()._cooccurrence_filter(text, triples, window=3)

    assert len(kept) == 1


def test_cooccurrence_filter_drops_genuinely_distant_pair():
    filler = "x " * 2000
    text = f"alpha {filler} beta"
    triples = [{"subject": "alpha", "relation": "REL", "object": "beta"}]

    kept = _verifier()._cooccurrence_filter(text, triples, window=3)

    assert kept == []


def test_cooccurrence_filter_passes_through_missing_endpoint():
    text = "alpha only appears here"
    triples = [{"subject": "alpha", "relation": "REL", "object": "beta"}]

    kept = _verifier()._cooccurrence_filter(text, triples, window=3)

    # beta absent -> defer to LLM verifier, do not drop deterministically
    assert len(kept) == 1


def test_merge_resolved_by_id_collapses_cross_batch_duplicates():
    resolved = [
        {
            "id": "metformin",
            "text": "Metformin",
            "labels": ["Metformin"],
            "mentions": ["Metformin"],
            "confidence": 0.6,
            "is_known_entity": False,
            "source_segments": ["seg1"],
        },
        {
            "id": "metformin",
            "text": "Metformin",
            "labels": ["metformin", "Glucophage"],
            "mentions": ["metformin", "Glucophage"],
            "confidence": 0.9,
            "is_known_entity": True,
            "source_segments": ["seg2"],
        },
    ]

    merged = EntityExtractor._merge_resolved_by_id(resolved)

    assert len(merged) == 1
    entity = merged[0]
    assert entity["confidence"] == 0.9  # max, not flattened or averaged
    assert entity["is_known_entity"] is True  # OR of flags
    assert set(entity["source_segments"]) == {"seg1", "seg2"}
    assert "Glucophage" in entity["mentions"]


def test_merge_resolved_by_id_preserves_distinct_entities():
    resolved = [
        {"id": "a", "text": "A", "confidence": 0.5},
        {"id": "b", "text": "B", "confidence": 0.7},
    ]

    merged = EntityExtractor._merge_resolved_by_id(resolved)

    assert [e["id"] for e in merged] == ["a", "b"]
