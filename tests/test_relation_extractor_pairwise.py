from multi_agent_kg.agents.base import AgentContext
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.core import DeliberativeOrchestrator, KnowledgeGraph, LLMConfig


def _make_extractor(**kwargs) -> RelationExtractor:
    kwargs.setdefault("enable_relation_gleaning", False)
    return RelationExtractor(
        knowledge_graph=KnowledgeGraph(),
        llm_config=LLMConfig(model="test-model"),
        use_self_consistency=False,
        **kwargs,
    )


def test_build_local_entity_pairs_limits_to_sentence_local_pairs() -> None:
    extractor = _make_extractor(enable_open_world=False)
    text = "CNN is used for classification. BLEU evaluates translation."
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
        {"id": "bleu", "text": "BLEU", "type": "Metric"},
        {"id": "translation", "text": "translation", "type": "Task"},
    ]

    pairs = extractor._build_local_entity_pairs(text, entities)
    pair_ids = {
        (pair["head_candidate_id"], pair["tail_candidate_id"], pair["sentence_index"])
        for pair in pairs
    }

    assert ("cnn", "classification", 0) in pair_ids
    assert ("bleu", "translation", 1) in pair_ids
    assert ("cnn", "translation", 0) not in pair_ids
    assert ("cnn", "translation", 1) not in pair_ids


def test_dedupe_triples_prefers_higher_confidence() -> None:
    extractor = _make_extractor(enable_open_world=False)
    triples = [
        {
            "subject": "CNN",
            "subject_id": "cnn",
            "relation": "Used-for",
            "object": "classification",
            "object_id": "classification",
            "confidence": 0.4,
        },
        {
            "subject": "CNN",
            "subject_id": "cnn",
            "relation": "Used-for",
            "object": "classification",
            "object_id": "classification",
            "confidence": 0.9,
        },
    ]

    deduped = extractor._dedupe_triples(triples)

    assert len(deduped) == 1
    assert deduped[0]["confidence"] == 0.9


def test_fixed_schema_confidence_floor_filters_weak_triples() -> None:
    extractor = _make_extractor(
        enable_open_world=False,
        fixed_schema_min_triple_confidence=0.75,
    )
    triples = [
        {
            "subject": "CNN",
            "subject_id": "cnn",
            "relation": "Used-for",
            "object": "classification",
            "object_id": "classification",
            "confidence": 0.74,
        },
        {
            "subject": "CNN",
            "subject_id": "cnn",
            "relation": "Used-for",
            "object": "image classification",
            "object_id": "image_classification",
            "confidence": 0.85,
        },
    ]

    filtered = extractor._filter_fixed_schema_triples_by_confidence(triples)

    assert len(filtered) == 1
    assert filtered[0]["object_id"] == "image_classification"


def test_pairwise_scoring_uses_exact_entity_pairs(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False)
    text = "CNN is used for classification."
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
    ]

    def fake_call_llm(*args, **kwargs):
        return {
            "predictions": [
                {
                    "pair_index": 0,
                    "relation": "Used-for",
                    "confidence": 0.88,
                    "evidence": "CNN is used for classification",
                }
            ]
        }

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)
    triples, stats = extractor._stage_pairwise_relation_scoring(
        text=text,
        entities=entities,
        allowed_relation_types=["Used-for", "Part-of"],
        stage1_relation_types=["Used-for"],
    )

    assert stats["pairwise_pairs_considered"] >= 1
    assert stats["pairwise_positive_predictions"] == 1
    assert len(triples) == 1
    assert triples[0]["subject_id"] == "cnn"
    assert triples[0]["object_id"] == "classification"
    assert triples[0]["relation"] == "Used-for"


def test_fixed_schema_pairwise_considers_full_schema_when_stage1_misses_relation(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False)
    text = "CNN is used for classification."
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
    ]
    captured = {}

    def fake_call_llm(*, prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "predictions": [
                {
                    "pair_index": 0,
                    "relation": "Used-for",
                    "confidence": 0.88,
                    "evidence": "CNN is used for classification",
                }
            ]
        }

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)
    triples, _ = extractor._stage_pairwise_relation_scoring(
        text=text,
        entities=entities,
        allowed_relation_types=["Used-for", "Part-of", "Conjunction"],
        stage1_relation_types=["Conjunction"],
    )

    assert "Used-for" in captured["prompt"]
    assert len(triples) == 1
    assert triples[0]["relation"] == "Used-for"


def test_run_activates_pairwise_path_in_fixed_schema_mode(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False, enable_fixed_schema_pairwise=True)
    context = AgentContext(document_id="doc1", text="CNN is used for classification.")
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
    ]
    pairwise_calls = {"count": 0}

    monkeypatch.setattr(
        extractor,
        "_stage1_identify_relations",
        lambda *args, **kwargs: [
            {"relation_type": "Used-for", "definition": "x", "count_in_text": 1}
        ],
    )
    monkeypatch.setattr(extractor, "_stage2_head_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(extractor, "_stage3_tail_binding", lambda *args, **kwargs: [])

    def fake_pairwise(*args, **kwargs):
        pairwise_calls["count"] += 1
        return (
            [
                {
                    "subject": "CNN",
                    "subject_id": "cnn",
                    "relation": "Used-for",
                    "object": "classification",
                    "object_id": "classification",
                    "confidence": 0.9,
                    "evidence": "CNN is used for classification.",
                }
            ],
            {
                "pairwise_pairs_considered": 1,
                "pairwise_positive_predictions": 1,
                "pairwise_triples_added": 1,
            },
        )

    monkeypatch.setattr(extractor, "_stage_pairwise_relation_scoring", fake_pairwise)

    result = extractor.run(
        context=context,
        entities=entities,
        domain_config={"relation_types": [{"type": "Used-for", "description": "used for"}]},
    )

    assert pairwise_calls["count"] == 1
    assert len(result.items) == 1
    assert result.metadata["pairwise_triples_added"] == 1


def test_run_ignores_malformed_stage1_relation_rows(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False, enable_fixed_schema_pairwise=False)
    context = AgentContext(document_id="doc1", text="CNN is used for classification.")
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
    ]
    monkeypatch.setattr(
        extractor,
        "_stage1_identify_relations",
        lambda *args, **kwargs: [{"bad_key": "Used-for"}, "junk", {"relation_type": ""}],
    )

    result = extractor.run(
        context=context,
        entities=entities,
        domain_config={"relation_types": [{"type": "Used-for", "description": "used for"}]},
    )

    assert result.items == []
    assert result.metadata["funnel_diagnostics"]["relations_found"] == 0


def test_fixed_schema_pairwise_runs_when_stage1_finds_no_relations(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False, enable_fixed_schema_pairwise=True)
    context = AgentContext(document_id="doc1", text="CNN is used for classification.")
    entities = [
        {"id": "cnn", "text": "CNN", "type": "Method"},
        {"id": "classification", "text": "classification", "type": "Task"},
    ]
    monkeypatch.setattr(extractor, "_stage1_identify_relations", lambda *args, **kwargs: [])
    monkeypatch.setattr(extractor, "_stage2_head_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(extractor, "_stage3_tail_binding", lambda *args, **kwargs: [])

    def fake_pairwise(*, stage1_relation_types, **kwargs):
        assert stage1_relation_types == ["Used-for"]
        return (
            [
                {
                    "subject": "CNN",
                    "subject_id": "cnn",
                    "relation": "Used-for",
                    "object": "classification",
                    "object_id": "classification",
                    "confidence": 0.9,
                    "evidence": "CNN is used for classification.",
                }
            ],
            {
                "pairwise_pairs_considered": 1,
                "pairwise_positive_predictions": 1,
                "pairwise_triples_added": 1,
            },
        )

    monkeypatch.setattr(extractor, "_stage_pairwise_relation_scoring", fake_pairwise)

    result = extractor.run(
        context=context,
        entities=entities,
        domain_config={"relation_types": [{"type": "Used-for", "description": "used for"}]},
    )

    assert len(result.items) == 1
    assert result.items[0]["relation"] == "Used-for"
    assert result.metadata["funnel_diagnostics"]["relations_found"] == 0
    assert result.metadata["pairwise_triples_added"] == 1


def test_run_keeps_segment_entities_after_coreference_metadata_loss(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False, enable_fixed_schema_pairwise=True)
    context = AgentContext(document_id="doc1", text="CNN is used for classification.")
    # This simulates post-coreference entities: the canonical text does not
    # appear literally in the segment, but a label/mention does.
    entities = [
        {
            "id": "cnn",
            "text": "convolutional neural network",
            "labels": ["CNN", "convolutional neural network"],
            "mentions": ["CNN"],
            "type": "Method",
        },
        {
            "id": "classification",
            "text": "text categorization",
            "labels": ["classification", "text categorization"],
            "mentions": ["classification"],
            "type": "Task",
        },
    ]
    seen_entity_counts = []

    def fake_stage1(text, segment_entities, *args, **kwargs):
        seen_entity_counts.append(len(segment_entities))
        return []

    def fake_pairwise(*, entities, stage1_relation_types, **kwargs):
        assert len(entities) == 2
        return (
            [
                {
                    "subject": "CNN",
                    "subject_id": "cnn",
                    "relation": "Used-for",
                    "object": "classification",
                    "object_id": "classification",
                    "confidence": 0.9,
                    "evidence": "CNN is used for classification.",
                }
            ],
            {
                "pairwise_pairs_considered": 1,
                "pairwise_positive_predictions": 1,
                "pairwise_triples_added": 1,
            },
        )

    monkeypatch.setattr(extractor, "_stage1_identify_relations", fake_stage1)
    monkeypatch.setattr(extractor, "_stage2_head_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(extractor, "_stage3_tail_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(extractor, "_stage_pairwise_relation_scoring", fake_pairwise)

    result = extractor.run(
        context=context,
        segments=[{"segment_id": "doc1_s0", "text": "CNN is used for classification."}],
        entities=entities,
        domain_config={"relation_types": [{"type": "Used-for", "description": "used for"}]},
    )

    assert seen_entity_counts == [2]
    assert len(result.items) == 1
    assert result.items[0]["subject_id"] == "cnn"


def test_relation_funnel_diagnostics_counts_stage_losses() -> None:
    extractor = _make_extractor(enable_open_world=False)
    diagnostics = extractor._new_funnel_diagnostics(document_id="doc1")

    extractor._record_funnel_segment(
        diagnostics,
        segment_id="s1",
        entities_seen=4,
        relations_found=2,
        head_bindings=3,
        tail_triples=2,
        pairwise_pairs=6,
        pairwise_positives=1,
        pairwise_triples=1,
        gleaned_triples=2,
        invalid_self_refs=1,
        post_align_triples=3,
        post_dedupe_triples=2,
    )

    assert diagnostics["segments_processed"] == 1
    assert diagnostics["entities_seen"] == 4
    assert diagnostics["relations_found"] == 2
    assert diagnostics["head_bindings"] == 3
    assert diagnostics["tail_triples"] == 2
    assert diagnostics["pairwise_pairs_considered"] == 6
    assert diagnostics["pairwise_positive_predictions"] == 1
    assert diagnostics["pairwise_triples_added"] == 1
    assert diagnostics["gleaned_triples_added"] == 2
    assert diagnostics["invalid_self_refs_filtered"] == 1
    assert diagnostics["post_alignment_triples"] == 3
    assert diagnostics["post_dedupe_triples"] == 2
    assert diagnostics["segment_summaries"][0]["segment_id"] == "s1"


def test_open_world_adds_deterministic_answer_value_attribute_triples(monkeypatch) -> None:
    extractor = _make_extractor(
        enable_open_world=True,
        enable_fixed_schema_pairwise=False,
        enable_deterministic_attribute_binding=True,
    )
    context = AgentContext(
        document_id="doc1",
        text="The Androscoggin Bank Colisee has a capacity of 3,677 seated.",
    )
    entities = [
        {
            "id": "androscoggin_bank_colisee",
            "text": "Androscoggin Bank Colisee",
            "labels": ["Androscoggin Bank Colisee"],
            "type": "Arena",
        },
        {
            "id": "3677_seated",
            "text": "3,677 seated",
            "labels": ["3,677 seated"],
            "type": "Quantity",
            "source_text": "The Androscoggin Bank Colisee has a capacity of 3,677 seated.",
        },
    ]
    monkeypatch.setattr(extractor, "_stage1_identify_relations", lambda *args, **kwargs: [])

    result = extractor.run(
        context=context,
        segments=[{"segment_id": "doc1_s0", "text": context.text}],
        entities=entities,
        domain_config={"relation_types": []},
    )

    assert any(item["relation"] == "HAS_CAPACITY" for item in result.items)
    triple = next(item for item in result.items if item["relation"] == "HAS_CAPACITY")
    assert triple["subject_id"] == "androscoggin_bank_colisee"
    assert triple["object_id"] == "3677_seated"
    assert triple["metadata"].get("typed_attribute_fact") is True


def test_open_world_binds_typed_attributes_without_biography_whitelists(monkeypatch) -> None:
    extractor = _make_extractor(
        enable_open_world=True,
        enable_fixed_schema_pairwise=False,
        enable_deterministic_attribute_binding=True,
    )
    context = AgentContext(
        document_id="doc1",
        text="Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    )
    entities = [
        {"id": "scott_derrickson", "text": "Scott Derrickson", "type": "Person"},
        {"id": "july_16_1966", "text": "July 16, 1966", "type": "Date"},
        {"id": "american", "text": "American", "type": "Nationality"},
        {"id": "director", "text": "director", "type": "Role"},
        {"id": "screenwriter", "text": "screenwriter", "type": "Role"},
        {"id": "producer", "text": "producer", "type": "Role"},
    ]
    monkeypatch.setattr(extractor, "_stage1_identify_relations", lambda *args, **kwargs: [])

    result = extractor.run(
        context=context,
        segments=[{"segment_id": "doc1_s0", "text": context.text}],
        entities=entities,
        domain_config={"relation_types": []},
    )

    triples = {
        (item["subject_id"], item["relation"], item["object_id"])
        for item in result.items
    }
    assert ("scott_derrickson", "BORN_ON_DATE", "july_16_1966") in triples
    assert ("scott_derrickson", "HAS_NATIONALITY", "american") in triples
    assert ("scott_derrickson", "EXERCISES_PROFESSIONAL_ROLE", "director") in triples
    assert ("scott_derrickson", "EXERCISES_PROFESSIONAL_ROLE", "screenwriter") in triples
    assert ("scott_derrickson", "EXERCISES_PROFESSIONAL_ROLE", "producer") in triples
    assert all(
        item["metadata"].get("typed_attribute_fact")
        for item in result.items
        if item["relation"] in {"BORN_ON_DATE", "HAS_NATIONALITY", "EXERCISES_PROFESSIONAL_ROLE"}
    )


def test_typed_attribute_binding_rejects_nearby_common_noun_subjects(monkeypatch) -> None:
    extractor = _make_extractor(
        enable_open_world=True,
        enable_fixed_schema_pairwise=False,
        enable_deterministic_attribute_binding=True,
    )
    context = AgentContext(
        document_id="doc1",
        text="Scott Derrickson (born July 16, 1966) is an American film producer.",
    )
    entities = [
        {"id": "scott_derrickson", "text": "Scott Derrickson", "type": "Person"},
        {"id": "july_16_1966", "text": "July 16, 1966", "type": "Date"},
        {"id": "american", "text": "American", "type": "Nationality"},
        {"id": "film_producer", "text": "film producer", "type": "Entity"},
    ]
    monkeypatch.setattr(extractor, "_stage1_identify_relations", lambda *args, **kwargs: [])

    result = extractor.run(
        context=context,
        segments=[{"segment_id": "doc1_s0", "text": context.text}],
        entities=entities,
        domain_config={"relation_types": []},
    )

    triples = {
        (item["subject_id"], item["relation"], item["object_id"])
        for item in result.items
    }
    assert ("scott_derrickson", "BORN_ON_DATE", "july_16_1966") in triples
    assert ("scott_derrickson", "HAS_NATIONALITY", "american") in triples
    assert ("film_producer", "HAS_NATIONALITY", "american") not in triples
    assert ("july_16_1966", "HAS_NATIONALITY", "american") not in triples


def test_orchestrator_summarizes_relation_funnel_diagnostics() -> None:
    orchestrator = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        knowledge_graph=KnowledgeGraph(),
        enable_governance=False,
    )
    orchestrator.processing_history = [
        {
            "document_id": "doc-a",
            "results": {
                "relation_funnel_diagnostics": {
                    "segments_processed": 1,
                    "entities_seen": 4,
                    "relations_found": 2,
                    "tail_triples": 0,
                    "post_alignment_triples": 0,
                    "post_dedupe_triples": 0,
                    "final_triples": 0,
                }
            },
        },
        {
            "document_id": "doc-b",
            "results": {
                "relation_funnel_diagnostics": {
                    "segments_processed": 2,
                    "entities_seen": 5,
                    "relations_found": 1,
                    "tail_triples": 1,
                    "post_alignment_triples": 1,
                    "post_dedupe_triples": 1,
                    "final_triples": 1,
                }
            },
        },
    ]

    summary = orchestrator.get_relation_funnel_summary()

    assert summary["documents_with_diagnostics"] == 2
    assert summary["segments_processed"] == 3
    assert summary["relations_found"] == 3
    assert summary["tail_triples"] == 1
    assert summary["docs_with_zero_final_triples"] == ["doc-a"]
    assert summary["docs_with_stage1_relations_but_no_final_triples"] == ["doc-a"]
    assert summary["tail_triples_per_stage1_relation"] == 0.3333


def test_gleaning_pass_filters_to_allowed_fixed_schema_relations(monkeypatch) -> None:
    extractor = _make_extractor(enable_open_world=False, enable_relation_gleaning=True)

    def fake_call_llm(*args, **kwargs):
        return {
            "missing_triples": [
                {
                    "subject": "CNN",
                    "subject_id": "cnn",
                    "relation": "Used-for",
                    "object": "classification",
                    "object_id": "classification",
                    "confidence": 0.7,
                    "evidence": "CNN is used for classification.",
                    "rationale": "missed relation",
                },
                {
                    "subject": "CNN",
                    "relation": "INVENTED",
                    "object": "classification",
                },
            ]
        }

    monkeypatch.setattr(extractor, "call_llm", fake_call_llm)

    triples = extractor._stage4_glean_missing_triples(
        text="CNN is used for classification.",
        entities=[
            {"id": "cnn", "text": "CNN", "type": "Method"},
            {"id": "classification", "text": "classification", "type": "Task"},
        ],
        relation_types=["Used-for"],
        existing_triples=[],
    )

    assert len(triples) == 1
    assert triples[0]["relation"] == "Used-for"
    assert triples[0]["metadata"]["gleaned"] is True
