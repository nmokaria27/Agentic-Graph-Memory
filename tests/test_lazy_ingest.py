"""Tests for lazy/hybrid ingestion (deferred query-driven extraction)."""

import hashlib

from multi_agent_kg.core.lazy_ingest import LazyIngestor, split_text


def _fake_embed(texts):
    """Deterministic pseudo-embeddings with mild lexical signal.

    Dimension i gets weight from hashed tokens, so shared words produce
    higher cosine similarity than disjoint ones.
    """
    dim = 64
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        vectors.append(vec)
    return vectors


def _ingestor(process_calls):
    def process_fn(text, document_id=None):
        process_calls.append({"text": text, "document_id": document_id})

    return LazyIngestor(
        process_fn=process_fn,
        embed_fn=_fake_embed,
        query_embed_fn=lambda q: _fake_embed([q])[0],
        chunk_size=200,
        chunk_overlap=20,
    )


CORPUS = (
    "Marie Curie discovered polonium and radium in Paris. "
    "She won two Nobel Prizes for her research on radioactivity. "
    + "Filler sentence about unrelated weather patterns and geology. " * 20
    + "The Beatles recorded Abbey Road in London in 1969. "
    "John Lennon and Paul McCartney wrote most of the songs."
)


def test_split_text_covers_whole_document():
    chunks = split_text(CORPUS, chunk_size=200, overlap=20)
    assert len(chunks) > 3
    assert chunks[0]["char_start"] == 0
    assert chunks[-1]["char_end"] >= len(CORPUS) - 1
    # Every chunk non-empty and within bounds
    assert all(c["text"] for c in chunks)


def test_ingest_is_cheap_no_extraction_calls():
    calls = []
    ingestor = _ingestor(calls)
    n = ingestor.ingest(CORPUS, document_id="doc1")
    assert n > 3
    assert calls == []  # no extraction on ingest
    assert ingestor.stats["chunks_indexed"] == n


def test_materialize_runs_extraction_on_relevant_chunks_only():
    calls = []
    ingestor = _ingestor(calls)
    ingestor.ingest(CORPUS, document_id="doc1")

    result = ingestor.materialize_for_query(
        "Who discovered polonium and radium?", top_k=2, max_chunks=2
    )
    assert result["materialized"] > 0
    assert len(calls) == 1
    assert "polonium" in calls[0]["text"].lower()
    assert calls[0]["document_id"].startswith("doc1__lazy_")
    # Only a fraction of the corpus was extracted
    assert result["materialized"] < len(ingestor.chunks)


def test_materialize_does_not_reprocess_chunks():
    calls = []
    ingestor = _ingestor(calls)
    ingestor.ingest(CORPUS, document_id="doc1")

    first = ingestor.materialize_for_query("Who discovered polonium?", top_k=2, max_chunks=2)
    second = ingestor.materialize_for_query("Who discovered polonium?", top_k=2, max_chunks=2)
    assert first["materialized"] > 0
    assert second["materialized"] == 0
    assert second["already_processed"] > 0
    assert len(calls) == 1


def test_persistence_roundtrip(tmp_path):
    calls = []
    ingestor = _ingestor(calls)
    ingestor.ingest(CORPUS, document_id="doc1")
    ingestor.materialize_for_query("Who discovered polonium?", top_k=2, max_chunks=2)
    ingestor.save_dir(str(tmp_path))

    restored = _ingestor(calls)
    assert restored.load_dir(str(tmp_path))
    assert restored.chunks.keys() == ingestor.chunks.keys()
    assert restored.processed == ingestor.processed
    # Restored index still answers queries without re-processing old chunks
    again = restored.materialize_for_query("Who discovered polonium?", top_k=2, max_chunks=2)
    assert again["materialized"] == 0


def test_materialize_empty_ingestor_is_noop():
    calls = []
    ingestor = _ingestor(calls)
    result = ingestor.materialize_for_query("anything")
    assert result == {"materialized": 0, "already_processed": 0}
    assert calls == []
