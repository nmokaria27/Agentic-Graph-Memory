"""EXP-SPGOV (GB-9) — governed_singlepass extraction mode.

Contract: the wide-harvest relationship candidates BECOME the triples (zero
RelationExtractor calls), evidence linking + deliberation are forced off, and
the candidates are converted/deduplicated into the standard triple shape.
"""
import pytest

from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig


def _orch(**kwargs):
    return DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        extraction_mode="governed_singlepass",
        enable_self_consistency=False,
        enable_open_world=False,
        enable_cross_document=False,
        enable_governance=True,
        **kwargs,
    )


def test_mode_forces_skip_flags_and_wide_front_end():
    orch = _orch()
    assert orch.extraction_mode == "governed_singlepass"
    assert orch.skip_evidence_linking is True
    assert orch.enable_deliberation is False
    # The entity extractor only understands deliberative/wide — it must get wide.
    assert orch.entity_extractor.extraction_mode == "wide"


def test_wide_candidates_to_triples_conversion():
    """The real stage-4 conversion helper: dedups case-insensitively, drops
    self-loops and malformed entries, defaults/repairs confidence."""
    triples = DeliberativeOrchestrator._wide_candidates_to_triples([
        {"source": "Marie Curie", "relation": "WON", "target": "Nobel Prize", "confidence": 0.9},
        {"source": "Marie Curie", "relation": "won", "target": "nobel prize"},  # dup (case)
        {"source": "Warsaw", "relation": "PART_OF", "target": "Warsaw"},        # self-loop
        {"source": "", "relation": "BORN_IN", "target": "Warsaw"},              # malformed
        "garbage string",                                                        # non-dict
        {"source": "Marie Curie", "relation": "BORN_IN", "target": "Warsaw",
         "confidence": "not-a-number"},                                          # bad conf
    ])

    assert len(triples) == 2
    assert {t["relation"] for t in triples} == {"WON", "BORN_IN"}
    assert all(t["subject"] == "Marie Curie" for t in triples)
    born = next(t for t in triples if t["relation"] == "BORN_IN")
    assert born["confidence"] == 0.7  # repaired default


def test_stage4_uses_candidates_and_never_calls_rhf(monkeypatch):
    """End-to-end through process_document up to stage 4: RelationExtractor
    must never run; the harvested candidates must land in context.relations."""
    orch = _orch()

    monkeypatch.setattr(
        orch.relation_extractor, "run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("RelationExtractor.run called in governed_singlepass mode")),
    )

    # Stage 1: one segment; Stage 2: fixed domain config; Stage 3: canned
    # entities + wide candidates (as the wide harvest would leave them);
    # stop the pipeline right after stage 4 via the connectivity gate.
    from multi_agent_kg.agents.base import ExtractionResult

    def _R(items, metadata=None, confidence=0.9):
        return ExtractionResult(items=items, confidence=confidence,
                                metadata=metadata or {})

    monkeypatch.setattr(orch.document_processor, "run",
                        lambda ctx, **k: _R([{"text": "Marie Curie won the Nobel Prize.",
                                              "segment_id": "s0"}]))
    monkeypatch.setattr(orch.domain_classifier, "run",
                        lambda ctx, **k: _R([{"primary_domain": "general",
                                              "entity_types": [], "relation_types": []}]))

    def _fake_entity_run(ctx, **kwargs):
        orch.entity_extractor.wide_relation_candidates = [
            {"source": "Marie Curie", "relation": "WON", "target": "Nobel Prize",
             "confidence": 0.9},
        ]
        return _R([{"id": "marie_curie", "text": "Marie Curie", "type": "Person"},
                   {"id": "nobel_prize", "text": "Nobel Prize", "type": "Award"}])

    monkeypatch.setattr(orch.entity_extractor, "run", _fake_entity_run)

    # Cut the run off AFTER stage 4. Important: verification's GB-1 degrade
    # path catches Exception and continues into stage 9 (LLM dedup) — a
    # sentinel raised from verification_agent.run is swallowed and the test
    # hangs on a real LLM call. Stop from knowledge_organizer instead, and
    # make verification a no-LLM pass-through so we never leave the process.
    class _Stop(Exception):
        pass

    def _fake_verify(ctx, **kwargs):
        triples = kwargs.get("triples") or getattr(ctx, "relations", []) or []
        entities = kwargs.get("entities") or getattr(ctx, "entities", []) or []
        return _R({"entities": entities, "approved_triples": triples,
                   "rejected_triples": []})

    monkeypatch.setattr(orch.verification_agent, "run", _fake_verify)
    monkeypatch.setattr(
        orch.knowledge_organizer, "run",
        lambda *a, **k: (_ for _ in ()).throw(_Stop("stage4 done")),
    )

    captured = {}
    real_convert = DeliberativeOrchestrator._wide_candidates_to_triples

    def _spy_convert(cands):
        out = real_convert(cands)
        captured["triples"] = out
        return out

    monkeypatch.setattr(DeliberativeOrchestrator, "_wide_candidates_to_triples",
                        staticmethod(_spy_convert))

    with pytest.raises(_Stop):
        orch.process_document(text="Marie Curie won the Nobel Prize.",
                              document_id="spgov_test_doc")

    assert captured.get("triples") == [
        {"subject": "Marie Curie", "relation": "WON", "object": "Nobel Prize",
         "confidence": 0.9},
    ]


def test_hybrid_wide_mode_unchanged():
    """Control: plain wide mode must NOT get the skip flags forced."""
    orch = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        extraction_mode="wide",
        enable_self_consistency=False,
        enable_governance=True,
    )
    assert orch.skip_evidence_linking is False
    assert orch.enable_deliberation is True
    assert orch.entity_extractor.extraction_mode == "wide"
