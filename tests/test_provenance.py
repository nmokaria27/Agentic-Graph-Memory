"""Unit tests for the unified provenance schema (core/provenance.py).

Pure-logic tests: no LLM, no KG. Cover ref construction, record assembly,
merge semantics across coref/dedup, attach-into-metadata, and audit read-back.
"""

from multi_agent_kg.core import provenance as prov


def test_source_ref_carries_span_and_truncates_snippet() -> None:
    ref = prov.source_ref(
        "doc_1",
        segment_id="doc_1_seg_2",
        chunk_index=2,
        char_start=100,
        char_end=140,
        source_path="/tmp/doc_1.txt",
        snippet="x" * 400,
    )
    assert ref["document_id"] == "doc_1"
    assert ref["chunk_index"] == 2
    assert ref["char_start"] == 100 and ref["char_end"] == 140
    assert len(ref["snippet"]) == 280  # snippet capped


def test_build_provenance_coerces_bad_confidence() -> None:
    p = prov.build_provenance(
        refs=[prov.source_ref("d", char_start=0, char_end=5)],
        extractor="EntityExtractor",
        confidence="not-a-number",  # type: ignore[arg-type]
    )
    assert p["confidence"] == 0.0
    assert p["extractor"] == "EntityExtractor"
    assert p["confidence_source"] == prov.CONFIDENCE_EXTRACTION
    assert p["evidence_sentences"] == []


def test_merge_unions_refs_and_keeps_max_confidence() -> None:
    a = prov.build_provenance(
        refs=[prov.source_ref("doc_1", segment_id="s0", char_start=0, char_end=4)],
        extractor="EntityExtractor",
        confidence=0.6,
        confidence_source=prov.CONFIDENCE_EXTRACTION,
    )
    b = prov.build_provenance(
        refs=[prov.source_ref("doc_1", segment_id="s3", char_start=90, char_end=98)],
        extractor="EntityExtractor",
        confidence=0.82,
        confidence_source=prov.CONFIDENCE_VERIFICATION,
    )
    merged = prov.merge(a, b)
    assert len(merged["refs"]) == 2  # both mentions retained
    assert merged["confidence"] == 0.82
    assert merged["confidence_source"] == prov.CONFIDENCE_VERIFICATION


def test_merge_dedupes_identical_refs() -> None:
    ref = prov.source_ref("doc_1", segment_id="s0", char_start=0, char_end=4)
    a = prov.build_provenance(refs=[ref], extractor="E", confidence=0.5)
    b = prov.build_provenance(refs=[dict(ref)], extractor="E", confidence=0.5)
    merged = prov.merge(a, b)
    assert len(merged["refs"]) == 1


def test_merge_handles_none_side() -> None:
    b = prov.build_provenance(
        refs=[prov.source_ref("d", char_start=1, char_end=2)],
        extractor="RelationExtractor",
        confidence=0.7,
    )
    merged = prov.merge(None, b)
    assert merged["confidence"] == 0.7
    assert merged["extractor"] == "RelationExtractor"


def test_attach_merges_when_provenance_exists() -> None:
    meta = {}
    p1 = prov.build_provenance(
        refs=[prov.source_ref("d", segment_id="s0", char_start=0, char_end=3)],
        extractor="E", confidence=0.4,
    )
    prov.attach(meta, p1)
    p2 = prov.build_provenance(
        refs=[prov.source_ref("d", segment_id="s1", char_start=10, char_end=13)],
        extractor="E", confidence=0.9,
    )
    prov.attach(meta, p2)
    stored = prov.get(meta)
    assert len(stored["refs"]) == 2
    assert stored["confidence"] == 0.9


def test_attach_preserves_other_metadata() -> None:
    meta = {"type": "PERSON", "aliases": ["x"]}
    prov.attach(meta, prov.build_provenance(
        refs=[prov.source_ref("d", char_start=0, char_end=1)],
        extractor="E", confidence=0.5,
    ))
    assert meta["type"] == "PERSON"
    assert meta["aliases"] == ["x"]
    assert prov.PROVENANCE_KEY in meta


def test_triple_provenance_carries_evidence() -> None:
    p = prov.build_provenance(
        refs=[prov.source_ref("doc_1", segment_id="s2", char_start=50, char_end=90)],
        extractor="RelationExtractor",
        confidence=0.88,
        evidence_sentences=["Metformin reduces HbA1c levels."],
    )
    assert p["evidence_sentences"] == ["Metformin reduces HbA1c levels."]


def test_describe_summarizes_for_audit() -> None:
    p = prov.build_provenance(
        refs=[prov.source_ref("doc_1", char_start=0, char_end=4)],
        extractor="EntityExtractor",
        confidence=0.91,
        confidence_source=prov.CONFIDENCE_VERIFICATION,
    )
    text = prov.describe({prov.PROVENANCE_KEY: p})
    assert "EntityExtractor" in text
    assert "0.91" in text
    assert "doc_1[0:4]" in text


def test_get_and_describe_handle_missing() -> None:
    assert prov.get(None) is None
    assert prov.get({}) is None
    assert prov.describe({}) == "<no provenance>"
