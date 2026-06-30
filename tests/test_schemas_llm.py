from multi_agent_kg.schemas_llm import (
    CoreferenceOut,
    EntityExtractionOut,
    EntityGroupOut,
    EntityOut,
    PredictionOut,
    TripleOut,
)


def test_confidence_string_coerced_to_float() -> None:
    e = EntityOut(text="Caroline", type="PERSON", confidence="0.8")
    assert e.confidence == 0.8


def test_confidence_clamped_to_unit_range() -> None:
    assert EntityOut(text="x", confidence=1.7).confidence == 1.0
    assert EntityOut(text="x", confidence=-0.2).confidence == 0.0


def test_confidence_uncoercible_falls_back() -> None:
    assert EntityOut(text="x", confidence="high").confidence == 0.5
    assert EntityOut(text="x", confidence=None).confidence == 0.5


def test_missing_type_defaults_to_unknown() -> None:
    assert EntityOut(text="x").type == "UNKNOWN"


def test_whitespace_stripped_from_strings() -> None:
    assert EntityOut(text="  Caroline  ").text == "Caroline"


def test_extra_keys_ignored_not_rejected() -> None:
    e = EntityOut(text="x", type="PERSON", confidence=0.9, hallucinated_field="junk")
    assert not hasattr(e, "hallucinated_field")
    assert e.text == "x"


def test_extraction_drops_broken_item_keeps_good_ones() -> None:
    raw = {
        "entities": [
            {"text": "Caroline", "type": "PERSON", "confidence": 0.9},
            {"text": "   ", "type": "PERSON"},          # empty -> dropped
            {"text": "Insulin", "confidence": "0.7"},   # coerced, type default
        ]
    }
    out = EntityExtractionOut.model_validate(raw)
    assert [e.text for e in out.entities] == ["Caroline", "Insulin"]
    assert out.entities[1].confidence == 0.7
    assert out.entities[1].type == "UNKNOWN"


def test_extraction_accepts_bare_top_level_list() -> None:
    raw = [{"text": "Caroline", "type": "PERSON"}]
    out = EntityExtractionOut.model_validate(raw)
    assert [e.text for e in out.entities] == ["Caroline"]


def test_extraction_empty_response_yields_empty_model() -> None:
    assert EntityExtractionOut().entities == []
    assert EntityExtractionOut.model_validate({}).entities == []


def test_coreference_parses_groups() -> None:
    raw = {
        "entity_groups": [
            {
                "canonical_id": "world_health_organization",
                "canonical_name": "World Health Organization",
                "type": "ORG",
                "mentions": ["WHO", "World Health Organization"],
                "is_known_entity": True,
            }
        ]
    }
    out = CoreferenceOut.model_validate(raw)
    assert out.entity_groups[0].canonical_id == "world_health_organization"
    assert out.entity_groups[0].is_known_entity is True


def test_coreference_group_defaults_are_safe() -> None:
    g = EntityGroupOut()
    assert g.canonical_id == ""
    assert g.type == "UNKNOWN"
    assert g.mentions == []
    assert g.is_known_entity is False


def test_coreference_empty_response_yields_empty_model() -> None:
    assert CoreferenceOut.model_validate({}).entity_groups == []


def test_bare_scalar_input_is_crash_proof() -> None:
    # The original line-611 crash: a thinking model returns raw CoT prose (a str)
    # instead of JSON. model_validate must degrade to empty, not raise.
    assert CoreferenceOut.model_validate("raw CoT prose, not JSON").entity_groups == []
    assert EntityExtractionOut.model_validate("still prose").entities == []
    assert EntityExtractionOut.model_validate(None).entities == []


# ── Relation / triple item models ─────────────────────────────────────


def test_prediction_coerces_confidence_and_pair_index() -> None:
    p = PredictionOut.model_validate(
        {"pair_index": "3", "relation": "USED-FOR", "confidence": "1.4"}
    )
    assert p.pair_index == 3       # str -> int
    assert p.confidence == 1.0     # clamped
    assert p.relation == "USED-FOR"


def test_prediction_keeps_none_relation_for_downstream_filtering() -> None:
    # "NONE"/empty must survive validation; the extractor decides to skip them.
    assert PredictionOut.model_validate({"relation": "NONE"}).relation == "NONE"
    assert PredictionOut.model_validate({}).relation == ""


def test_prediction_junk_pair_index_degrades_to_none() -> None:
    assert PredictionOut.model_validate({"pair_index": "abc"}).pair_index is None


def test_prediction_preserves_unknown_keys_no_yield_loss() -> None:
    # extra="allow": pipeline/LLM extras must round-trip through model_dump.
    dumped = PredictionOut.model_validate(
        {"relation": "PART-OF", "sentence_index": 7, "custom": "keep me"}
    ).model_dump()
    assert dumped["sentence_index"] == 7
    assert dumped["custom"] == "keep me"


def test_triple_partial_fields_survive_with_defaults() -> None:
    t = TripleOut.model_validate({"subject": "metformin", "object": "diabetes"})
    assert t.subject == "metformin"
    assert t.object == "diabetes"
    assert t.relation == ""        # downstream guard handles empty
    assert t.confidence == 0.5


def test_triple_preserves_alias_and_metadata_keys() -> None:
    # relation_type/predicate aliases + metadata are read downstream; must persist.
    dumped = TripleOut.model_validate(
        {
            "subject": "a",
            "object": "b",
            "relation_type": "treats",
            "metadata": {"pairwise_scored": True},
            "confidence": "0.9",
        }
    ).model_dump()
    assert dumped["relation_type"] == "treats"
    assert dumped["metadata"] == {"pairwise_scored": True}
    assert dumped["confidence"] == 0.9
