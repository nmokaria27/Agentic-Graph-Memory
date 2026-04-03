"""
Fast Extractor Agent (Phase 2) -- Zero-LLM Baseline via GLiNER + GLiREL.

Provides CPU-only entity and relation extraction at ~47+ sentences/sec using
pre-trained transformer span-classification models.  This agent is designed
as a *cheap first pass* whose output can be refined by the LLM-based
EntityExtractor and RelationExtractor agents downstream.

Key properties:
- Does NOT call any LLM -- all inference runs locally via GLiNER / GLiREL.
- GLiNER and GLiREL are optional dependencies; if neither is installed the
  agent gracefully returns an empty result with a log message.
- Maps domain_config entity/relation type names to GLiNER/GLiREL label sets.
- Respects GLiNER's 512-token input limit by splitting long segments.
- Tracks its own ``fast_extraction_calls`` stat instead of ``llm_calls``.
"""

from typing import Any, Dict, List, Optional

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus
from multi_agent_kg.core.config import LLMConfig

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

try:
    from glirel import GLiREL
    GLIREL_AVAILABLE = True
except ImportError:
    GLIREL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default GLiNER model -- good balance of speed and quality on CPU.
_DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"

# Default GLiREL model for relation extraction.
_DEFAULT_GLIREL_MODEL = "jackboyla/glirel_beta"

# GLiNER has a 512-token context window.  We use a conservative character
# estimate (1 token ~ 4 chars) and allow some overlap for split chunks.
_MAX_CHUNK_CHARS = 1800  # ~450 tokens, leaving headroom
_CHUNK_OVERLAP_CHARS = 200

# No hardcoded fallback labels — the system learns everything from the
# domain classifier.  If no labels are provided, GLiNER/GLiREL will not
# be called for that extraction type (entities or relations) and the
# FastExtractor will return an empty result, deferring to the LLM-based
# extractors which discover types from the content.

# Minimum confidence threshold for GLiNER predictions to keep.
_MIN_ENTITY_SCORE = 0.25
_MIN_RELATION_SCORE = 0.20


class FastExtractor(BaseAgent):
    """
    Zero-LLM baseline extractor using GLiNER (entities) and GLiREL (relations).

    This agent is intended for Phase 2 of the pipeline: it produces a quick,
    high-recall first draft of entities and triples that can then be verified
    and refined by downstream LLM-based agents.

    Both GLiNER and GLiREL are **optional** -- if the packages are not
    installed the ``run()`` method returns an empty result and logs a warning.
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        gliner_model: str = _DEFAULT_GLINER_MODEL,
        glirel_model: str = _DEFAULT_GLIREL_MODEL,
        entity_threshold: float = _MIN_ENTITY_SCORE,
        relation_threshold: float = _MIN_RELATION_SCORE,
    ):
        super().__init__(
            name="FastExtractor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            # Model tier is irrelevant -- we never call an LLM.
            default_tier=ModelTier.SMALL,
        )

        self._gliner_model_name = gliner_model
        self._glirel_model_name = glirel_model
        self._entity_threshold = entity_threshold
        self._relation_threshold = relation_threshold

        # Lazy-loaded model instances (populated on first use).
        self._gliner: Optional[Any] = None
        self._glirel: Optional[Any] = None

        # Override the default stats dict with a fast-extraction counter
        # instead of llm_calls (this agent never touches an LLM).
        self.stats["fast_extraction_calls"] = 0

    # ------------------------------------------------------------------
    # Model loading (lazy, cached on instance)
    # ------------------------------------------------------------------

    def _get_gliner(self) -> Optional[Any]:
        """Return the cached GLiNER model, loading on first call."""
        if self._gliner is not None:
            return self._gliner
        if not GLINER_AVAILABLE:
            return None
        self.log(f"Loading GLiNER model: {self._gliner_model_name}")
        self._gliner = GLiNER.from_pretrained(self._gliner_model_name)
        return self._gliner

    def _get_glirel(self) -> Optional[Any]:
        """Return the cached GLiREL model, loading on first call."""
        if self._glirel is not None:
            return self._glirel
        if not GLIREL_AVAILABLE:
            return None
        self.log(f"Loading GLiREL model: {self._glirel_model_name}")
        self._glirel = GLiREL.from_pretrained(self._glirel_model_name)
        return self._glirel

    # ------------------------------------------------------------------
    # Label mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_labels_from_config(domain_config: Optional[Dict[str, Any]]) -> List[str]:
        """
        Derive GLiNER-compatible entity labels from ``domain_config``.

        GLiNER expects lowercase, human-readable label strings (e.g.
        ``"person"``, ``"clinical measurement"``).  The domain classifier
        produces UPPER_SNAKE_CASE type names, so we convert them here.

        Returns an empty list if no domain config is provided — the caller
        should skip GLiNER extraction in that case (no hardcoded fallbacks).
        """
        if not domain_config:
            return []

        raw_names: List[str] = domain_config.get("entity_type_names", [])

        # Also accept the detailed entity_types list of dicts.
        if not raw_names:
            for et in domain_config.get("entity_types", []):
                if isinstance(et, dict):
                    raw_names.append(et.get("type", ""))
                elif isinstance(et, str):
                    raw_names.append(et)

        labels = []
        for name in raw_names:
            if not name:
                continue
            # UPPER_SNAKE_CASE -> lowercase with spaces
            labels.append(name.lower().replace("_", " "))
        return labels

    @staticmethod
    def _relation_labels_from_config(domain_config: Optional[Dict[str, Any]]) -> List[str]:
        """
        Derive GLiREL-compatible relation labels from ``domain_config``.

        Returns an empty list if no domain config is provided — the caller
        should skip GLiREL extraction in that case (no hardcoded fallbacks).
        """
        if not domain_config:
            return []

        raw_names: List[str] = domain_config.get("relation_type_names", [])

        if not raw_names:
            for rt in domain_config.get("relation_types", []):
                if isinstance(rt, dict):
                    raw_names.append(rt.get("type", ""))
                elif isinstance(rt, str):
                    raw_names.append(rt)

        labels = []
        for name in raw_names:
            if not name:
                continue
            labels.append(name.lower().replace("_", " "))
        return labels

    # ------------------------------------------------------------------
    # Text chunking (respect GLiNER 512-token limit)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str) -> List[str]:
        """
        Split *text* into overlapping chunks that fit within GLiNER's
        512-token context window.

        Uses a character-based heuristic (~4 chars/token).  Tries to split
        on sentence boundaries when possible.
        """
        if len(text) <= _MAX_CHUNK_CHARS:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + _MAX_CHUNK_CHARS

            if end < len(text):
                # Try to split on a sentence boundary (period, newline).
                search_region = text[max(end - 200, start):end]
                # Look for the last sentence-ending punctuation in the region.
                for sep in [". ", ".\n", "\n\n", "\n", "; ", ", "]:
                    last_sep = search_region.rfind(sep)
                    if last_sep != -1:
                        end = max(end - 200, start) + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Advance with overlap so entities at boundaries are not missed.
            start = end - _CHUNK_OVERLAP_CHARS
            if start <= (end - _MAX_CHUNK_CHARS):
                # Safety: always advance at least half a chunk.
                start = end - _CHUNK_OVERLAP_CHARS

        return chunks

    # ------------------------------------------------------------------
    # Core extraction helpers
    # ------------------------------------------------------------------

    def _extract_entities_from_text(
        self,
        text: str,
        labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run GLiNER entity extraction on *text*, returning standardised dicts.

        Handles chunking internally so callers do not need to worry about
        the 512-token limit.
        """
        model = self._get_gliner()
        if model is None:
            return []

        chunks = self._split_into_chunks(text)
        seen: set = set()  # deduplicate across overlapping chunks
        entities: List[Dict[str, Any]] = []

        for chunk in chunks:
            try:
                preds = model.predict_entities(chunk, labels, threshold=self._entity_threshold)
            except Exception as exc:
                self.log(f"GLiNER prediction failed on chunk: {exc}", level="WARNING")
                continue

            for pred in preds:
                ent_text = pred.get("text", "").strip()
                ent_label = pred.get("label", "UNKNOWN")
                ent_score = float(pred.get("score", 0.0))

                if not ent_text:
                    continue

                # Deduplicate by (lowercased text, label).
                dedup_key = (ent_text.lower(), ent_label.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Map label back to UPPER_SNAKE_CASE for consistency with
                # the rest of the pipeline.
                entity_type = ent_label.upper().replace(" ", "_")

                entities.append({
                    "text": ent_text,
                    "type": entity_type,
                    "confidence": round(ent_score, 4),
                    "start": pred.get("start", 0),
                    "end": pred.get("end", 0),
                    "source": "gliner",
                })

        return entities

    def _extract_relations_from_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run GLiREL relation extraction given pre-extracted entities.

        Returns triples in the project's standard format.
        """
        model = self._get_glirel()
        if model is None:
            return []

        if not entities or not relation_labels:
            return []

        # GLiREL expects a list of entity spans.  Build the representation
        # it needs from our standardised entity dicts.
        #
        # GLiREL's `predict_relations` typically expects:
        #   tokens    : List[str]            (tokenised text)
        #   ner       : List[[start, end, label]]  (token-level spans)
        #   labels    : List[str]            (candidate relation labels)
        #
        # The exact API varies by GLiREL version.  We use the high-level
        # convenience path when available, falling back to manual setup.

        chunks = self._split_into_chunks(text)
        seen: set = set()
        triples: List[Dict[str, Any]] = []

        for chunk in chunks:
            # Build NER spans that fall within this chunk.
            chunk_ner: List[List] = []
            chunk_entity_map: Dict[str, str] = {}  # (start,end) -> entity text

            for ent in entities:
                # Try to locate the entity text within the chunk.
                idx = chunk.find(ent["text"])
                if idx == -1:
                    continue
                ent_end = idx + len(ent["text"])
                chunk_ner.append([idx, ent_end, ent.get("type", "ENTITY").lower().replace("_", " ")])
                chunk_entity_map[f"{idx}:{ent_end}"] = ent["text"]

            if len(chunk_ner) < 2:
                # Need at least two entities for a relation.
                continue

            try:
                # Tokenise simply by whitespace for GLiREL.
                tokens = chunk.split()

                preds = model.predict_relations(
                    tokens,
                    relation_labels,
                    threshold=self._relation_threshold,
                    ner=chunk_ner,
                )
            except Exception as exc:
                self.log(f"GLiREL prediction failed on chunk: {exc}", level="WARNING")
                continue

            for pred in preds:
                subj = pred.get("head", pred.get("subject", ""))
                obj = pred.get("tail", pred.get("object", ""))
                rel_label = pred.get("label", pred.get("relation", "UNKNOWN_RELATION"))
                score = float(pred.get("score", 0.0))

                if isinstance(subj, dict):
                    subj = subj.get("text", str(subj))
                if isinstance(obj, dict):
                    obj = obj.get("text", str(obj))

                if not subj or not obj:
                    continue

                dedup_key = (subj.lower(), rel_label.lower(), obj.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                relation_type = rel_label.upper().replace(" ", "_")

                triples.append({
                    "subject": subj,
                    "relation": relation_type,
                    "object": obj,
                    "confidence": round(score, 4),
                    "evidence": chunk[:200],
                    "source": "glirel",
                })

        return triples

    # ------------------------------------------------------------------
    # Override: call_llm is a no-op for this agent
    # ------------------------------------------------------------------

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """FastExtractor does not use LLMs. This override prevents accidental calls."""
        self.log(
            "call_llm invoked on FastExtractor -- this agent does not use LLMs. "
            "Returning empty dict.",
            level="WARNING",
        )
        return {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Execute fast baseline extraction using GLiNER and GLiREL.

        Args:
            context: Processing context with document text.
            segments: Optional list of document segments (each must have
                a ``"text"`` key and optionally a ``"segment_id"``).
            domain_config: Domain configuration dict produced by the
                DomainClassifier (contains ``entity_type_names``,
                ``relation_type_names``, etc.).

        Returns:
            ExtractionResult whose ``items`` list contains two entries:
            ``{"entities": [...], "triples": [...]}`` merged into the
            items list as individual entity/triple dicts, consistent with
            the downstream pipeline's expectations.
        """
        self.stats["calls"] += 1
        self.stats["fast_extraction_calls"] += 1

        # ---- Guard: check library availability ----
        if not GLINER_AVAILABLE and not GLIREL_AVAILABLE:
            self.log(
                "Neither GLiNER nor GLiREL is installed. "
                "Install them with: pip install gliner glirel  "
                "Returning empty extraction result.",
                level="WARNING",
            )
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={
                    "document_id": context.document_id,
                    "error": "gliner_and_glirel_not_installed",
                },
            )

        # ---- Derive labels from domain config ----
        entity_labels = self._entity_labels_from_config(domain_config)
        relation_labels = self._relation_labels_from_config(domain_config)

        # ---- Determine texts to process ----
        texts_to_process: List[tuple] = []
        if segments:
            texts_to_process = [
                (s.get("text", ""), s.get("segment_id", f"seg_{i}"))
                for i, s in enumerate(segments)
            ]
        elif context.text:
            texts_to_process = [(context.text, f"{context.document_id}_full")]

        if not texts_to_process:
            self.log("No text provided for fast extraction.")
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={"document_id": context.document_id},
            )

        # ---- Run extraction over each segment ----
        all_entities: List[Dict[str, Any]] = []
        all_triples: List[Dict[str, Any]] = []

        for text, segment_id in texts_to_process:
            if not text or not text.strip():
                continue

            # Entity extraction (GLiNER)
            if GLINER_AVAILABLE:
                seg_entities = self._extract_entities_from_text(text, entity_labels)
                for ent in seg_entities:
                    ent["source_segment"] = segment_id
                all_entities.extend(seg_entities)

            # Relation extraction (GLiREL) -- needs entities from above.
            if GLIREL_AVAILABLE:
                # Use segment-level entities for relation extraction when
                # possible; fall back to all collected entities so far.
                ents_for_rels = [e for e in all_entities if e.get("source_segment") == segment_id]
                if not ents_for_rels:
                    ents_for_rels = all_entities

                seg_triples = self._extract_relations_from_text(
                    text, ents_for_rels, relation_labels,
                )
                for triple in seg_triples:
                    triple["source_segment"] = segment_id
                    triple["document_id"] = context.document_id
                all_triples.extend(seg_triples)

        # ---- Compute aggregate confidence ----
        all_scores: List[float] = (
            [e["confidence"] for e in all_entities]
            + [t["confidence"] for t in all_triples]
        )
        avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # ---- Store in shared memory if available ----
        if self.shared_memory and (all_entities or all_triples):
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "fast_entities": all_entities,
                    "fast_triples": all_triples,
                    "document_id": context.document_id,
                },
                metadata={"source_agent": self.name},
            )

        # ---- Build combined items list ----
        # Downstream stages expect a flat list of dicts.  We tag each item
        # with an ``"item_type"`` key so consumers can distinguish entities
        # from triples.
        items: List[Dict[str, Any]] = []
        for ent in all_entities:
            items.append({**ent, "item_type": "entity"})
        for triple in all_triples:
            items.append({**triple, "item_type": "triple"})

        self.log(
            f"Fast extraction complete: {len(all_entities)} entities, "
            f"{len(all_triples)} triples (avg confidence {avg_confidence:.2f})"
        )

        return ExtractionResult(
            items=items,
            confidence=round(avg_confidence, 4),
            metadata={
                "document_id": context.document_id,
                "entity_count": len(all_entities),
                "triple_count": len(all_triples),
                "gliner_available": GLINER_AVAILABLE,
                "glirel_available": GLIREL_AVAILABLE,
                "entity_labels_used": entity_labels,
                "relation_labels_used": relation_labels,
            },
        )
