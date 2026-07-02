"""
Lazy / hybrid ingestion (LazyGraphRAG-style deferred extraction).

The full 9-agent pipeline costs ~minutes per 20K chars, which makes
million-character corpora infeasible to ingest eagerly. LazyIngestor splits
the corpus into chunks and embeds them (near-zero LLM cost). The expensive
extraction pipeline runs later, at query time, only on the top query-relevant
chunks that have not been processed yet — so the knowledge graph grows
incrementally in the regions questions actually touch.

Usage:
    ingestor = LazyIngestor(process_fn=orchestrator.process_document)
    ingestor.ingest(huge_text, document_id="doc1")     # cheap
    ingestor.materialize_for_query("who founded X?")   # extracts top chunks
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from multi_agent_kg.core.vector_index import VectorIndex


def split_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Char-window chunks aligned to paragraph/sentence boundaries where possible."""
    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer to break at a paragraph, else a sentence, inside the
            # last 40% of the window.
            floor = start + int(chunk_size * 0.6)
            cut = text.rfind("\n\n", floor, end)
            if cut == -1:
                cut = text.rfind(". ", floor, end)
                if cut != -1:
                    cut += 1  # keep the period
            if cut != -1 and cut > start:
                end = cut
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"char_start": start, "char_end": end, "text": chunk_text})
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


class LazyIngestor:
    """Chunk-index a corpus cheaply; extract KG only where queries need it.

    `process_fn(text, document_id)` is the expensive extraction entry point
    (normally DeliberativeOrchestrator.process_document). Keeping it as a
    callable keeps this module orchestrator-agnostic and testable.
    """

    def __init__(
        self,
        process_fn: Callable[..., Any],
        embedding_model: Optional[str] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        query_embed_fn: Optional[Callable[[str], List[float]]] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        self.process_fn = process_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.processed: set = set()
        self.index = VectorIndex(
            model=embedding_model, embed_fn=embed_fn, query_embed_fn=query_embed_fn
        )
        self.stats = {
            "documents_ingested": 0,
            "chunks_indexed": 0,
            "queries_served": 0,
            "chunks_materialized": 0,
            "materialize_calls": 0,
        }

    # -- ingestion (cheap) --------------------------------------------------

    def ingest(self, text: str, document_id: str) -> int:
        """Chunk + embed a document without running extraction. Returns chunk count."""
        pieces = split_text(text, self.chunk_size, self.chunk_overlap)
        items: Dict[str, str] = {}
        for i, piece in enumerate(pieces):
            chunk_id = f"{document_id}::chunk_{i}"
            self.chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "index": i,
                "char_start": piece["char_start"],
                "char_end": piece["char_end"],
                "text": piece["text"],
            }
            items[chunk_id] = piece["text"]
        if items:
            self.index.upsert(items)
        self.stats["documents_ingested"] += 1
        self.stats["chunks_indexed"] += len(items)
        return len(items)

    # -- materialization (expensive, query-driven) ---------------------------

    def materialize_for_query(
        self,
        query: str,
        top_k: int = 8,
        max_chunks: int = 6,
    ) -> Dict[str, Any]:
        """Run extraction on the top query-relevant unprocessed chunks.

        Retrieves top_k chunks by embedding similarity, keeps the unprocessed
        ones (up to max_chunks), groups them per source document in reading
        order, and hands each group to process_fn as one mini-document.
        """
        self.stats["queries_served"] += 1
        if not self.chunks:
            return {"materialized": 0, "already_processed": 0}
        try:
            hits = self.index.search(query, top_k=top_k)
        except Exception:
            hits = []
        selected = [cid for cid, _ in hits if cid in self.chunks and cid not in self.processed]
        already = sum(1 for cid, _ in hits if cid in self.processed)
        selected = selected[:max_chunks]
        if not selected:
            return {"materialized": 0, "already_processed": already}

        # Group per document, in document order, so extraction sees coherent text.
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for cid in selected:
            chunk = self.chunks[cid]
            by_doc.setdefault(chunk["document_id"], []).append(chunk)

        self.stats["materialize_calls"] += 1
        materialized = 0
        for document_id, doc_chunks in by_doc.items():
            doc_chunks.sort(key=lambda c: c["index"])
            text = "\n\n".join(c["text"] for c in doc_chunks)
            batch_id = f"{document_id}__lazy_{self.stats['materialize_calls']}"
            self.process_fn(text, document_id=batch_id)
            for chunk in doc_chunks:
                self.processed.add(chunk["chunk_id"])
                materialized += 1
        self.stats["chunks_materialized"] += materialized
        return {"materialized": materialized, "already_processed": already}

    # -- persistence ----------------------------------------------------------

    def save_dir(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.index.save(os.path.join(directory, "chunks.npz"))
        state = {
            "chunks": self.chunks,
            "processed": sorted(self.processed),
            "stats": self.stats,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(os.path.join(directory, "lazy_state.json"), "w", encoding="utf-8") as handle:
            json.dump(state, handle)

    def load_dir(self, directory: str) -> bool:
        """Restore chunk store + processed set. Returns False when absent."""
        state_path = os.path.join(directory, "lazy_state.json")
        if not os.path.exists(state_path):
            return False
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        self.chunks = state.get("chunks", {})
        self.processed = set(state.get("processed", []))
        self.stats.update(state.get("stats", {}))
        index = VectorIndex.load(
            os.path.join(directory, "chunks.npz"),
            model=self.index.model,
            embed_fn=self.index._embed_fn,
            query_embed_fn=self.index._query_embed_fn,
        )
        if index is not None:
            self.index = index
        return True


__all__ = ["LazyIngestor", "split_text"]
