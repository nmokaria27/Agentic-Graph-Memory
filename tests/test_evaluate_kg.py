from evaluation.evaluate_kg import (
    compute_degeneracy_rate,
    compute_rogue_entity_stats,
    evaluate_entities_strict,
    evaluate_entities_partial,
    evaluate_triples_fuzzy,
    evaluate_triples_strict,
)


def test_entities_partial_uses_label_when_text_missing() -> None:
    gold = [{"text": "Japanese text", "type": "Material"}]
    pred = [{"id": "japanese_text", "labels": ["Japanese text"], "type": "MATERIAL"}]

    prf, halluc = evaluate_entities_partial(gold, pred)

    assert prf.tp == 1
    assert prf.fp == 0
    assert prf.fn == 0
    assert halluc == 0.0


def test_entities_strict_uses_label_when_text_missing() -> None:
    gold = [{"text": "Japanese text", "type": "Material"}]
    pred = [{"id": "japanese_text", "labels": ["Japanese text"], "type": "MATERIAL"}]

    prf, _, _, halluc = evaluate_entities_strict(gold, pred)

    assert prf.tp == 1
    assert prf.fp == 0
    assert prf.fn == 0
    assert halluc == 0.0


def test_triple_matching_uses_original_surface_forms_from_metadata() -> None:
    gold = [
        {
            "subject": "Japanese text",
            "relation": "Part-of",
            "object": "morphological analysis",
        }
    ]
    pred = [
        {
            "subject": "japanese_text",
            "relation": "Part-of",
            "object": "morphological_analysis",
            "metadata": {
                "original_subject": "Japanese text",
                "original_object": "morphological analysis",
            },
        }
    ]

    strict_prf, _, strict_halluc = evaluate_triples_strict(gold, pred)
    fuzzy_prf, fuzzy_halluc = evaluate_triples_fuzzy(gold, pred)

    assert strict_prf.tp == 1
    assert strict_prf.fp == 0
    assert strict_prf.fn == 0
    assert strict_halluc == 0.0

    assert fuzzy_prf.tp == 1
    assert fuzzy_prf.fp == 0
    assert fuzzy_prf.fn == 0
    assert fuzzy_halluc == 0.0


def test_degeneracy_metric_flags_bad_triples() -> None:
    triples = [
        {"subject": "IL-6", "relation": "ASSOCIATED_WITH", "object": "IL-6"},
        {"subject": "Apple Watch", "relation": "RELATED_TO", "object": "Apple Watch Series"},
        {"subject": "the study", "relation": "MENTIONS", "object": "HbA1c"},
        {"subject": "Metformin", "relation": "REDUCES", "object": "HbA1c"},
    ]

    stats = compute_degeneracy_rate(triples)

    assert stats["degenerate_count"] == 3
    assert stats["by_type"]["SELF_REF"] == 1
    assert stats["by_type"]["CONTAINMENT"] == 1
    assert stats["by_type"]["GENERIC_NODE"] == 1


def test_rogue_entity_metric_flags_unresolved_generic_orphans() -> None:
    entities = [
        {"id": "study", "text": "the study", "type": "GENERIC"},
        {"id": "rogue", "text": "made-up node", "type": "UNRESOLVED"},
        {"id": "il6", "text": "IL-6", "type": "BIOMARKER"},
    ]
    triples = [{"subject": "il6", "relation": "ASSOCIATED_WITH", "object": "rogue"}]

    stats = compute_rogue_entity_stats(entities, triples)

    assert stats["rogue_entity_count"] == 2
    assert stats["by_type"]["GENERIC_ENTITY"] == 1
    assert stats["by_type"]["UNRESOLVED"] == 1
    assert stats["by_type"]["ORPHAN_ENTITY"] == 1
