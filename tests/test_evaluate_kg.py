from evaluation.evaluate_kg import (
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
