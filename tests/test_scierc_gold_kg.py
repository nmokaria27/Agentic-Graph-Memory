from evaluation.adapters.scierc_gold_kg import build_scierc_gold_kg, canonical_entity_id


def test_canonical_entity_id() -> None:
    assert canonical_entity_id("IL-6") == "il_6"
    assert canonical_entity_id("  Neural Machine Translation ") == "neural_machine_translation"


def test_build_scierc_gold_kg_smoke() -> None:
    kg, org_chart, stats = build_scierc_gold_kg(
        ["evaluation/datasets/scierc/dev.json"],
        skip_generic=True,
    )

    assert stats["num_entities"] > 0
    assert stats["num_triples"] > 0
    assert stats["num_domains"] > 0
    assert "methods_and_models" in stats["domain_sizes"]
    assert len(kg.entities) == stats["num_entities"]
    assert len(kg.triples) == stats["num_triples"]
    assert len(org_chart.domains) == stats["num_domains"]
