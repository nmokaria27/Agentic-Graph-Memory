"""
Vector index layer for hybrid (graph + embedding) retrieval.

Design: vectors find semantic entry points (entity linking, triple evidence,
domain routing); the knowledge graph supplies multi-hop structure and
auditability. The index is derived state — always rebuildable from the
governed KG, never a source of truth.

Search is exact brute-force cosine over a normalized numpy matrix. At the
scale of these graphs (10^2-10^4 nodes) this is sub-millisecond and beats
any ANN library on simplicity and recall. Revisit (FAISS/LanceDB) only if
corpora grow past ~100k vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from multi_agent_kg.llm.openai_client import (
    DEFAULT_EMBEDDING_MODEL,
    embed_passages,
    embed_query,
)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """A minimal exact-search embedding index with content-hash caching."""

    def __init__(
        self,
        model: Optional[str] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        query_embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.model = model or DEFAULT_EMBEDDING_MODEL
        self._embed_fn = embed_fn
        self._query_embed_fn = query_embed_fn
        self.ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None  # (n, dim) float32, L2-normalized
        self.hashes: Dict[str, str] = {}
        self._id_to_row: Dict[str, int] = {}

    # -- embedding hooks (overridable for tests) -----------------------

    def _embed_passages(self, texts: List[str]) -> List[List[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        return embed_passages(texts, model=self.model)

    def _embed_query(self, text: str) -> List[float]:
        if self._query_embed_fn is not None:
            return self._query_embed_fn(text)
        if self._embed_fn is not None:
            return self._embed_fn([text])[0]
        return embed_query(text, model=self.model)

    # -- mutation -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._id_to_row

    def upsert(self, items: Dict[str, str]) -> int:
        """Insert or update items {id: text}. Re-embeds only changed texts.

        Returns the number of items actually (re-)embedded.
        """
        changed_ids: List[str] = []
        changed_texts: List[str] = []
        for item_id, text in items.items():
            digest = _text_hash(text)
            if self.hashes.get(item_id) == digest:
                continue
            changed_ids.append(item_id)
            changed_texts.append(text)
            self.hashes[item_id] = digest

        if not changed_ids:
            return 0

        vectors = np.asarray(self._embed_passages(changed_texts), dtype=np.float32)
        vectors = _normalize_rows(vectors)

        if self.matrix is None:
            self.matrix = vectors
            self.ids = list(changed_ids)
            self._id_to_row = {item_id: row for row, item_id in enumerate(self.ids)}
            return len(changed_ids)

        new_rows: List[np.ndarray] = []
        for item_id, vector in zip(changed_ids, vectors):
            row = self._id_to_row.get(item_id)
            if row is not None:
                self.matrix[row] = vector
            else:
                self._id_to_row[item_id] = len(self.ids) + len(new_rows)
                new_rows.append(vector)
        # Appended ids must be tracked alongside their rows
        appended = [i for i in changed_ids if self._id_to_row[i] >= len(self.ids)]
        if new_rows:
            self.matrix = np.vstack([self.matrix, np.asarray(new_rows, dtype=np.float32)])
            self.ids.extend(appended)
        return len(changed_ids)

    def remove(self, item_ids: List[str]) -> None:
        doomed = {i for i in item_ids if i in self._id_to_row}
        if not doomed:
            return
        keep_rows = [row for row, item_id in enumerate(self.ids) if item_id not in doomed]
        self.ids = [self.ids[row] for row in keep_rows]
        self.matrix = self.matrix[keep_rows] if self.matrix is not None and keep_rows else None
        if self.matrix is not None and len(self.ids) == 0:
            self.matrix = None
        self._id_to_row = {item_id: row for row, item_id in enumerate(self.ids)}
        for item_id in doomed:
            self.hashes.pop(item_id, None)

    # -- search ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Return [(id, cosine_score)] for the query text, best first."""
        if not self.ids:
            return []
        vector = np.asarray(self._embed_query(query), dtype=np.float32)
        return self.search_vec(vector, top_k=top_k, min_score=min_score)

    def search_vec(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        if self.matrix is None or not self.ids:
            return []
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        vector = vector / norm
        scores = self.matrix @ vector
        top_k = min(top_k, len(self.ids))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            (self.ids[row], float(scores[row]))
            for row in order
            if float(scores[row]) >= min_score
        ]

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        matrix = self.matrix if self.matrix is not None else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(
            path,
            ids=np.array(self.ids, dtype=object),
            matrix=matrix,
            hashes=np.array(json.dumps(self.hashes), dtype=object),
            model=np.array(self.model, dtype=object),
        )

    @classmethod
    def load(cls, path: str, model: Optional[str] = None, **kwargs: Any) -> Optional["VectorIndex"]:
        """Load an index. Returns None if missing or built with another model."""
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        stored_model = str(data["model"])
        expected = model or DEFAULT_EMBEDDING_MODEL
        if stored_model != expected:
            return None
        index = cls(model=stored_model, **kwargs)
        index.ids = [str(i) for i in data["ids"].tolist()]
        matrix = data["matrix"]
        index.matrix = matrix.astype(np.float32) if matrix.size else None
        index.hashes = json.loads(str(data["hashes"]))
        index._id_to_row = {item_id: row for row, item_id in enumerate(index.ids)}
        return index


def triple_key(triple: Any) -> str:
    return f"{triple.subject}|{triple.relation}|{triple.object}"


def verbalize_triple(triple: Any) -> str:
    text = (
        f"{triple.subject.replace('_', ' ')} "
        f"{triple.relation.replace('_', ' ').lower()} "
        f"{triple.object.replace('_', ' ')}"
    )
    evidence = ""
    metadata = getattr(triple, "metadata", None) or {}
    if isinstance(metadata, dict):
        evidence = str(metadata.get("evidence") or "").strip()
    return f"{text}. {evidence}".strip()


class KGVectorStore:
    """Entity, triple, and domain indices over one governed knowledge graph."""

    ENTITY_CONTEXT_TRIPLES = 8

    def __init__(
        self,
        model: Optional[str] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        query_embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        index_kwargs = {"model": model, "embed_fn": embed_fn, "query_embed_fn": query_embed_fn}
        self._index_kwargs = index_kwargs
        self.entity_index = VectorIndex(**index_kwargs)
        self.triple_index = VectorIndex(**index_kwargs)
        self.domain_index = VectorIndex(**index_kwargs)
        self._governed_kg: Any = None
        self._dirty_entities: Set[str] = set()
        self._dirty_triples: Set[str] = set()
        self._domains_dirty = False

    # -- text builders ----------------------------------------------------

    @staticmethod
    def _adjacency(kg: Any) -> Dict[str, List[Any]]:
        adj: Dict[str, List[Any]] = {}
        for triple in kg.triples:
            adj.setdefault(triple.subject, []).append(triple)
            adj.setdefault(triple.object, []).append(triple)
        return adj

    def _entity_text(
        self,
        kg: Any,
        entity_id: str,
        adjacency: Optional[Dict[str, List[Any]]] = None,
    ) -> str:
        entity = kg.entities.get(entity_id)
        labels = " ; ".join(entity.labels) if entity and entity.labels else ""
        entity_type = entity.type if entity and entity.type else ""
        neighbours = (
            adjacency.get(entity_id, [])
            if adjacency is not None
            else [t for t in kg.triples if t.subject == entity_id or t.object == entity_id]
        )
        context = [
            verbalize_triple(triple)
            for triple in neighbours[: self.ENTITY_CONTEXT_TRIPLES]
        ]
        parts = [entity_id.replace("_", " "), labels, entity_type, " | ".join(context)]
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _domain_text(domain: Any) -> str:
        relations = ", ".join(sorted(domain.relation_schema.keys())) if domain.relation_schema else ""
        memory_card = ""
        summary_fn = getattr(domain, "memory_card_summary", None)
        if callable(summary_fn):
            memory_card = summary_fn() or ""
        parts = [domain.label, domain.description, f"Relations: {relations}" if relations else "", memory_card]
        return ". ".join(part for part in parts if part)

    # -- build / refresh ---------------------------------------------------

    def build(self, governed_kg: Any) -> None:
        """Full (re)build of all three indices from a governed KG."""
        self._governed_kg = governed_kg
        kg = governed_kg.kg

        adjacency = self._adjacency(kg)
        self.entity_index.upsert(
            {entity_id: self._entity_text(kg, entity_id, adjacency) for entity_id in kg.entities}
        )
        self.triple_index.upsert(
            {triple_key(triple): verbalize_triple(triple) for triple in kg.triples}
        )
        self.domain_index.upsert(
            {domain.domain_id: self._domain_text(domain) for domain in governed_kg.org_chart.domains}
        )
        self._dirty_entities.clear()
        self._dirty_triples.clear()
        self._domains_dirty = False

    def mark_dirty(
        self,
        entity_ids: Optional[List[str]] = None,
        triples: Optional[List[Any]] = None,
        domains: bool = False,
    ) -> None:
        if entity_ids:
            self._dirty_entities.update(entity_ids)
        if triples:
            for triple in triples:
                self._dirty_triples.add(triple_key(triple))
                # 1-hop context of both endpoints changed too
                self._dirty_entities.add(triple.subject)
                self._dirty_entities.add(triple.object)
        if domains:
            self._domains_dirty = True

    def refresh(self) -> int:
        """Re-embed only dirty items. Returns count of re-embedded items."""
        if self._governed_kg is None:
            return 0
        kg = self._governed_kg.kg
        count = 0
        if self._dirty_entities:
            adjacency = self._adjacency(kg)
            items = {
                entity_id: self._entity_text(kg, entity_id, adjacency)
                for entity_id in self._dirty_entities
                if entity_id in kg.entities
            }
            count += self.entity_index.upsert(items)
            self._dirty_entities.clear()
        if self._dirty_triples:
            by_key = {triple_key(t): t for t in kg.triples}
            items = {
                key: verbalize_triple(by_key[key])
                for key in self._dirty_triples
                if key in by_key
            }
            count += self.triple_index.upsert(items)
            self._dirty_triples.clear()
        if self._domains_dirty:
            count += self.domain_index.upsert(
                {
                    domain.domain_id: self._domain_text(domain)
                    for domain in self._governed_kg.org_chart.domains
                }
            )
            self._domains_dirty = False
        return count

    # -- persistence ---------------------------------------------------------

    @staticmethod
    def _kg_content_hash(governed_kg: Any) -> str:
        kg = governed_kg.kg
        payload = json.dumps(
            {
                "entities": sorted(kg.entities.keys()),
                "triples": sorted(triple_key(t) for t in kg.triples),
            },
            sort_keys=True,
        )
        return _text_hash(payload)

    def save_dir(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.entity_index.save(os.path.join(directory, "entities.npz"))
        self.triple_index.save(os.path.join(directory, "triples.npz"))
        self.domain_index.save(os.path.join(directory, "domains.npz"))
        meta = {
            "model": self.entity_index.model,
            "kg_hash": self._kg_content_hash(self._governed_kg) if self._governed_kg else None,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    @classmethod
    def load_dir(
        cls,
        directory: str,
        governed_kg: Any,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional["KGVectorStore"]:
        """Load a store if it matches the KG content hash; else return None."""
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        store = cls(model=model, **kwargs)
        if meta.get("kg_hash") != cls._kg_content_hash(governed_kg):
            return None
        entity_index = VectorIndex.load(os.path.join(directory, "entities.npz"), model=model, **kwargs)
        triple_index = VectorIndex.load(os.path.join(directory, "triples.npz"), model=model, **kwargs)
        domain_index = VectorIndex.load(os.path.join(directory, "domains.npz"), model=model, **kwargs)
        if entity_index is None or triple_index is None or domain_index is None:
            return None
        store.entity_index = entity_index
        store.triple_index = triple_index
        store.domain_index = domain_index
        store._governed_kg = governed_kg
        return store


def rrf_fuse(
    rankings: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[str]:
    """Reciprocal-rank fusion of multiple ranked id lists."""
    weights = weights or [1.0] * len(rankings)
    scores: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


__all__ = [
    "VectorIndex",
    "KGVectorStore",
    "rrf_fuse",
    "triple_key",
    "verbalize_triple",
]
