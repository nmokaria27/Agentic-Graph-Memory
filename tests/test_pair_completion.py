"""GB-3 (EXP-PAIR-COMPLETE): pair-completion pass for co-occurring unlinked pairs.

Contract: candidates = extracted entity pairs whose surfaces co-occur within
the char window and have no existing triple between them (either direction,
id or surface); nearest-first, capped. Stage 4c is DEFAULT OFF.
"""
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig

TEXT = ("Marie Curie studied at the University of Paris. "
        "Pierre Curie married Marie Curie. "
        + "filler sentence. " * 40 +
        "Warsaw is far away in this text.")

ENTS = [
    {"id": "marie_curie", "text": "Marie Curie", "type": "PERSON"},
    {"id": "university_of_paris", "text": "University of Paris", "type": "ORG"},
    {"id": "pierre_curie", "text": "Pierre Curie", "type": "PERSON"},
    {"id": "warsaw", "text": "Warsaw", "type": "LOC"},
]


def test_candidates_window_linked_exclusion_and_cap():
    triples = [{"subject": "Marie Curie", "subject_id": "marie_curie",
                "relation": "SPOUSE_OF",
                "object": "Pierre Curie", "object_id": "pierre_curie"}]
    cands = RelationExtractor._pair_completion_candidates(
        TEXT, ENTS, triples, window_chars=300, max_pairs=100)
    pairs = {frozenset((a["id"], b["id"])) for a, b, _ in cands}
    # Unlinked near pairs present:
    assert frozenset(("marie_curie", "university_of_paris")) in pairs
    assert frozenset(("pierre_curie", "university_of_paris")) in pairs
    # Already-linked pair excluded:
    assert frozenset(("marie_curie", "pierre_curie")) not in pairs
    # Far entity (outside window) excluded:
    assert not any("warsaw" in p for p in pairs)
    # Cap respected + nearest-first ordering:
    capped = RelationExtractor._pair_completion_candidates(
        TEXT, ENTS, [], window_chars=10_000, max_pairs=2)
    assert len(capped) == 2
    assert capped[0][2] <= capped[1][2]


def test_extract_pair_completion_tags_source_and_filters(monkeypatch):
    ex = RelationExtractor.__new__(RelationExtractor)
    prompts = []

    def fake_call_llm(**kwargs):
        prompts.append(kwargs["prompt"])
        return {"triples": [
            {"subject": "Marie Curie", "subject_id": "marie_curie",
             "relation": "STUDIED_AT",
             "object": "University of Paris", "object_id": "university_of_paris",
             "confidence": 0.9, "evidence": "studied at"},
            {"subject": "Marie Curie", "relation": "SELF",
             "object": "Marie Curie"},  # degenerate — dropped
        ]}

    ex.call_llm = fake_call_llm
    out = ex.extract_pair_completion_relations(TEXT, ENTS, [])
    assert len(out) == 1
    assert out[0]["source"] == "pair_completion"
    assert out[0]["relation"] == "STUDIED_AT"
    # The candidate pairs (not raw entities) are what the prompt lists.
    assert "<-?->" in prompts[0]


def test_stage_4c_default_off():
    orch = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        enable_self_consistency=False,
        enable_governance=True,
    )
    assert orch.enable_pair_completion is False
    on = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        enable_self_consistency=False,
        enable_governance=True,
        enable_pair_completion=True,
    )
    assert on.enable_pair_completion is True
