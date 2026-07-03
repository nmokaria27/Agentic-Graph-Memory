"""Wide-harvest extraction mode (fused pipeline front end).

One singlepass-style call replaces the multi-stage per-segment extraction;
entities must come back in the standard {text, type, confidence} shape and
relationship candidates must be stashed for downstream seeding.
"""
from multi_agent_kg.agents.entity_extractor import EntityExtractor


def _wide_extractor(llm_response):
    ex = EntityExtractor.__new__(EntityExtractor)
    ex.extraction_mode = "wide"
    ex.wide_relation_candidates = []
    ex.call_llm = lambda *a, **k: llm_response
    return ex


def test_wide_segment_parses_entities_and_stashes_relations():
    ex = _wide_extractor({
        "entities": [
            {"text": "Ramey Idriss", "type": "PERSON", "confidence": 0.95},
            {"text": "The Woody Woodpecker Song", "type": "WORK", "confidence": "0.8"},
            {"text": "", "type": "JUNK"},  # empty text must be dropped
        ],
        "relationships": [
            {"source": "Ramey Idriss", "relation": "composed", "target": "The Woody Woodpecker Song"},
            {"source": "", "relation": "broken", "target": "x"},  # missing source dropped
        ],
    })
    typed = ex._extract_wide_segment("some text", ["PERSON", "WORK"])
    assert [e["text"] for e in typed] == ["Ramey Idriss", "The Woody Woodpecker Song"]
    assert typed[0]["confidence"] == 0.95
    assert typed[1]["confidence"] == 0.8  # string coerced by schema
    assert len(ex.wide_relation_candidates) == 1
    assert ex.wide_relation_candidates[0]["relation"] == "composed"


def test_wide_segment_survives_garbage_output():
    ex = _wide_extractor("reasoning prose, not JSON")
    assert ex._extract_wide_segment("text", []) == []
    assert ex.wide_relation_candidates == []


def test_wide_segment_no_relationships_key():
    ex = _wide_extractor({"entities": [{"text": "Alpha", "type": "ORG"}]})
    typed = ex._extract_wide_segment("text", [])
    assert [e["text"] for e in typed] == ["Alpha"]
    assert ex.wide_relation_candidates == []
