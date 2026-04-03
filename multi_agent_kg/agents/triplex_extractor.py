"""
Triplex Extractor Agent (Phase 6) -- Parallel KG Triplet Extraction via SciPhi Triplex.

Uses the SciPhi Triplex model (3.8B params) on Ollama for schema-guided
knowledge graph triplet extraction.  Triplex is purpose-built for KG
extraction: given entity types and predicates it returns structured
(subject, relation, object) triplets.

This agent runs **in parallel** with the main LLM-based extraction
pipeline to produce an independent set of entities and triples.  The
SchemaAligner agent downstream merges and reconciles the two result
sets.

Key properties:
- OPTIONAL agent -- if ``sciphi/triplex`` is not available on Ollama
  the agent gracefully returns empty results instead of crashing.
- Model availability is checked lazily on first use and cached in
  the class attribute ``_model_available``.
- Uses ``BaseAgent.call_llm()`` with ``tier=ModelTier.SMALL`` so that
  the AGENT_MODEL_OVERRIDES mechanism can route to ``sciphi/triplex``.
- Converts Triplex's output format to the project's standard entity
  and triple dicts (same schema as EntityExtractor / RelationExtractor).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

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
# Triplex prompt template
# ---------------------------------------------------------------------------

TRIPLEX_PROMPT = (
    "Perform Named Entity Recognition (NER) and extract knowledge graph "
    "triplets from the text. Entity Types: {types}. Predicates: {predicates}. "
    "Text: {text}"
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriplexExtractor(BaseAgent):
    """
    Parallel extraction sub-agent powered by the SciPhi Triplex model.

    Triplex (``sciphi/triplex``, 3.8B params) is a small, purpose-built
    model for schema-guided KG triplet extraction.  It accepts a list of
    entity types and predicates along with a text passage and returns
    structured triplets.

    This agent is **optional**: if the Triplex model is not pulled into
    Ollama the agent returns empty results without raising.

    The output is merged with the main pipeline's extractions by the
    downstream SchemaAligner agent to boost recall and confidence.

    Uses SharedMemory to:
    - Store Triplex-specific extraction results for downstream merging

    Uses MessageBus to:
    - Send extracted entities/triples to SchemaAligner
    """

    # Lazy-checked flag: None = not yet checked, True/False = cached result.
    _model_available: Optional[bool] = None

    # The Ollama model tag for Triplex.
    TRIPLEX_MODEL = "sciphi/triplex"

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.70,
    ):
        super().__init__(
            name="TriplexExtractor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.SMALL,
            quality_threshold=quality_threshold,
        )

    # ------------------------------------------------------------------
    # Model availability check
    # ------------------------------------------------------------------

    def _check_model_available(self) -> bool:
        """Check whether ``sciphi/triplex`` is available on Ollama.

        The result is cached on the **class** so that all instances share
        the outcome and only one HTTP request is ever made.
        """
        if TriplexExtractor._model_available is not None:
            return TriplexExtractor._model_available

        try:
            import httpx

            ollama_base = self._get_ollama_base()
            resp = httpx.get(f"{ollama_base}/api/tags", timeout=10.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Match both "sciphi/triplex" and "sciphi/triplex:latest" etc.
            available = any(
                n == self.TRIPLEX_MODEL or n.startswith(f"{self.TRIPLEX_MODEL}:")
                for n in model_names
            )
            TriplexExtractor._model_available = available
            if not available:
                self.log(
                    f"Triplex model '{self.TRIPLEX_MODEL}' not found on Ollama. "
                    f"Available models: {model_names}. "
                    f"TriplexExtractor will return empty results.",
                    level="WARNING",
                )
            else:
                self.log(f"Triplex model '{self.TRIPLEX_MODEL}' is available on Ollama.")
            return available
        except Exception as exc:
            self.log(
                f"Could not query Ollama for model availability: {exc}. "
                f"TriplexExtractor will return empty results.",
                level="WARNING",
            )
            TriplexExtractor._model_available = False
            return False

    @staticmethod
    def _get_ollama_base() -> str:
        """Return the Ollama base URL without the ``/v1`` suffix."""
        import os
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_entity_types(domain_config: Optional[Dict[str, Any]]) -> str:
        """Extract entity type names from domain_config and format for Triplex."""
        if not domain_config:
            return ""

        names: List[str] = domain_config.get("entity_type_names", [])
        if not names:
            for et in domain_config.get("entity_types", []):
                if isinstance(et, dict):
                    names.append(et.get("type", ""))
                elif isinstance(et, str):
                    names.append(et)

        # Triplex expects comma-separated entity type labels.
        return ", ".join(n for n in names if n)

    @staticmethod
    def _format_predicates(domain_config: Optional[Dict[str, Any]]) -> str:
        """Extract relation type names from domain_config and format for Triplex."""
        if not domain_config:
            return ""

        names: List[str] = domain_config.get("relation_type_names", [])
        if not names:
            for rt in domain_config.get("relation_types", []):
                if isinstance(rt, dict):
                    names.append(rt.get("type", ""))
                elif isinstance(rt, str):
                    names.append(rt)

        return ", ".join(n for n in names if n)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_triplex_response(
        self,
        response: Any,
        segment_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse Triplex model output into standard entity and triple dicts.

        Triplex may return its results in several formats:
        - A JSON object with ``entities`` and ``triples``/``triplets`` keys
        - A plain-text list of triplets (``<subject> | <predicate> | <object>``)
        - A JSON dict directly from ``call_llm`` (already parsed)

        This method handles all variants and normalises to the project's
        standard format.

        Returns:
            Tuple of (entities, triples).
        """
        entities: List[Dict[str, Any]] = []
        triples: List[Dict[str, Any]] = []
        seen_entities: set = set()

        # ------ Handle dict (already parsed JSON from call_llm) ------
        if isinstance(response, dict):
            raw_entities = response.get("entities", [])
            raw_triples = response.get("triples", response.get("triplets", []))

            for ent in raw_entities:
                if isinstance(ent, dict):
                    text = ent.get("text", ent.get("name", ent.get("entity", "")))
                    etype = ent.get("type", ent.get("entity_type", "UNKNOWN"))
                elif isinstance(ent, str):
                    text = ent
                    etype = "UNKNOWN"
                else:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                key = text.lower()
                if key not in seen_entities:
                    seen_entities.add(key)
                    entities.append({
                        "text": text,
                        "type": etype.upper().replace(" ", "_") if etype else "UNKNOWN",
                        "confidence": float(ent.get("confidence", 0.65)) if isinstance(ent, dict) else 0.65,
                        "source": "triplex",
                        "source_segment": segment_id,
                    })

            for triple in raw_triples:
                if isinstance(triple, dict):
                    subj = triple.get("subject", triple.get("head", ""))
                    rel = triple.get("relation", triple.get("predicate", triple.get("type", "")))
                    obj = triple.get("object", triple.get("tail", ""))
                    conf = float(triple.get("confidence", 0.65))
                    evidence = triple.get("evidence", "")
                elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
                    conf = 0.65
                    evidence = ""
                else:
                    continue

                subj = str(subj).strip()
                rel = str(rel).strip()
                obj = str(obj).strip()
                if not subj or not rel or not obj:
                    continue

                triples.append({
                    "subject": subj,
                    "relation": rel.upper().replace(" ", "_"),
                    "object": obj,
                    "confidence": conf,
                    "evidence": evidence,
                    "source": "triplex",
                    "source_segment": segment_id,
                })

                # Collect entities from triples as well
                for mention in (subj, obj):
                    key = mention.lower()
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append({
                            "text": mention,
                            "type": "UNKNOWN",
                            "confidence": conf * 0.9,
                            "source": "triplex",
                            "source_segment": segment_id,
                        })

            return entities, triples

        # ------ Handle string (raw text from Triplex) ------
        if isinstance(response, str):
            return self._parse_triplex_text(response, segment_id)

        return entities, triples

    def _parse_triplex_text(
        self,
        text: str,
        segment_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse plain-text Triplex output (pipe-delimited triplets).

        Triplex sometimes outputs lines like::

            Entity1 | PREDICATE | Entity2

        This method extracts those and converts them to standard format.
        """
        entities: List[Dict[str, Any]] = []
        triples: List[Dict[str, Any]] = []
        seen_entities: set = set()

        # Try JSON parse first in case the string is actually JSON.
        try:
            parsed = json.loads(text)
            return self._parse_triplex_response(parsed, segment_id)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to line-by-line pipe-delimited parsing.
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Match "subject | predicate | object" patterns
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                subj, rel, obj = parts[0], parts[1], parts[2]
                if not subj or not rel or not obj:
                    continue

                triples.append({
                    "subject": subj,
                    "relation": rel.upper().replace(" ", "_"),
                    "object": obj,
                    "confidence": 0.60,
                    "evidence": "",
                    "source": "triplex",
                    "source_segment": segment_id,
                })

                for mention in (subj, obj):
                    key = mention.lower()
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append({
                            "text": mention,
                            "type": "UNKNOWN",
                            "confidence": 0.55,
                            "source": "triplex",
                            "source_segment": segment_id,
                        })

        return entities, triples

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def _extract_segment(
        self,
        text: str,
        entity_types: str,
        predicates: str,
        segment_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run Triplex extraction on a single text segment.

        Returns:
            Tuple of (entities, triples).
        """
        prompt = TRIPLEX_PROMPT.format(
            types=entity_types or "Any",
            predicates=predicates or "Any",
            text=text,
        )

        try:
            response = self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are a knowledge graph extraction model. "
                    "Extract entities and relationships as structured triplets."
                ),
                tier=ModelTier.SMALL,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as exc:
            self.log(f"Triplex LLM call failed: {exc}", level="WARNING")
            return [], []

        return self._parse_triplex_response(response, segment_id)

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
        Extract entities and triples using the Triplex model.

        If the Triplex model is not available on Ollama this method
        returns an empty ``ExtractionResult`` without error.

        Args:
            context: Processing context with document text.
            segments: Optional list of document segments (each with a
                ``"text"`` key and optionally ``"segment_id"``).
            domain_config: Domain configuration dict from the
                DomainClassifier (provides entity types and relation
                types used to guide Triplex).

        Returns:
            ExtractionResult whose ``items`` contain entity and triple
            dicts tagged with ``item_type`` and ``source: "triplex"``.
        """
        self.stats["calls"] += 1

        # ---- Guard: check model availability ----
        if not self._check_model_available():
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={
                    "document_id": context.document_id,
                    "skipped": True,
                    "reason": "triplex_model_not_available",
                },
            )

        # ---- Derive schema labels ----
        entity_types = self._format_entity_types(domain_config)
        predicates = self._format_predicates(domain_config)

        # ---- Determine texts to process ----
        texts: List[Tuple[str, str]] = []
        if segments:
            texts = [
                (s.get("text", ""), s.get("segment_id", f"seg_{i}"))
                for i, s in enumerate(segments)
            ]
        elif context.text:
            texts = [(context.text, f"{context.document_id}_full")]

        if not texts:
            self.log("No text provided for Triplex extraction.")
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={"document_id": context.document_id},
            )

        # ---- Run extraction over each segment ----
        all_entities: List[Dict[str, Any]] = []
        all_triples: List[Dict[str, Any]] = []

        for text, segment_id in texts:
            if not text or not text.strip():
                continue

            seg_entities, seg_triples = self._extract_segment(
                text, entity_types, predicates, segment_id,
            )
            all_entities.extend(seg_entities)
            all_triples.extend(seg_triples)

        # ---- Compute aggregate confidence ----
        all_scores = (
            [e.get("confidence", 0.5) for e in all_entities]
            + [t.get("confidence", 0.5) for t in all_triples]
        )
        avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # ---- Store in shared memory ----
        if self.shared_memory and (all_entities or all_triples):
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "triplex_entities": all_entities,
                    "triplex_triples": all_triples,
                    "document_id": context.document_id,
                },
                metadata={"source_agent": self.name},
            )

        # ---- Build combined items list ----
        items: List[Dict[str, Any]] = []
        for ent in all_entities:
            items.append({**ent, "item_type": "entity"})
        for triple in all_triples:
            items.append({**triple, "item_type": "triple"})

        self.log(
            f"Triplex extraction complete: {len(all_entities)} entities, "
            f"{len(all_triples)} triples (avg confidence {avg_confidence:.2f})"
        )

        return ExtractionResult(
            items=items,
            confidence=round(avg_confidence, 4),
            metadata={
                "document_id": context.document_id,
                "entity_count": len(all_entities),
                "triple_count": len(all_triples),
                "model": self.TRIPLEX_MODEL,
                "entity_types_used": entity_types,
                "predicates_used": predicates,
            },
        )
