from multi_agent_kg.agents.base import AgentContext
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.core import KnowledgeGraph, LLMConfig


def _make_extractor(**kwargs) -> RelationExtractor:
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
