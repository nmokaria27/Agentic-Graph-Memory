"""
Schema Aligner Agent -- Merge and Normalise Multi-Source Extraction Results.

This agent sits downstream of the main LLM extraction pipeline and the
optional Triplex extraction pipeline.  It receives entities and triples
from both sources and produces a single, deduplicated, schema-normalised
result set.

Merge strategy:
1. **Entity deduplication**: fuzzy text matching (rapidfuzz token-sort-ratio
   when available, simple string comparison fallback).  When two entities
   match, the higher-confidence entry is kept and its metadata is enriched
   with the other source's aliases.
2. **Triple deduplication**: case-insensitive (subject, relation, object)
   matching.  When both sources agree on a triple the confidence is boosted.
3. **Type normalisation**: entity types and relation names are normalised
   to UPPER_SNAKE_CASE for consistency across downstream consumers.
4. **Ambiguous type resolution**: when two sources disagree on the entity
   type for a matched entity, the LLM is optionally consulted to pick the
   best type.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

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

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType

# ---------------------------------------------------------------------------
# Optional dependency: rapidfuzz
# ---------------------------------------------------------------------------
try:
    from rapidfuzz.fuzz import token_sort_ratio
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# LLM prompt for ambiguous entity-type resolution
# ---------------------------------------------------------------------------

TYPE_RESOLUTION_PROMPT = """You are a knowledge-graph schema expert.

Two extraction sources disagree on the type of an entity.

Entity text: "{entity_text}"
Source A type: "{type_a}"
Source B type: "{type_b}"

Domain context: {domain}

Pick the **single best** entity type for this entity.  You may choose one of
the two proposed types, or suggest a more accurate type if both are wrong.
Normalise the type to UPPER_SNAKE_CASE (e.g. PERSON, ORGANISATION, WORK_OF_ART).

Respond with JSON:
{{"resolved_type": "<TYPE>", "reason": "<one sentence explanation>"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def _to_upper_snake_case(name: str) -> str:
    """Convert an arbitrary type/relation string to UPPER_SNAKE_CASE.

    Examples
    --------
    >>> _to_upper_snake_case("workOfArt")
    'WORK_OF_ART'
    >>> _to_upper_snake_case("organisation")
    'ORGANISATION'
    >>> _to_upper_snake_case("has-part")
    'HAS_PART'
    >>> _to_upper_snake_case("LOCATED_IN")
    'LOCATED_IN'
    """
    if not name:
        return name

    # Replace hyphens and dots with underscores
    s = name.replace("-", "_").replace(".", "_")

    # Insert underscores before uppercase runs in camelCase / PascalCase
    # e.g. "workOfArt" -> "work_Of_Art"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Handle sequences of uppercase letters followed by lowercase
    # e.g. "HTMLParser" -> "HTML_Parser"
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)

    # Collapse multiple underscores and strip leading/trailing ones
    s = re.sub(r"_+", "_", s).strip("_")

    return s.upper()


def _fuzzy_score(a: str, b: str) -> float:
    """Return a fuzzy similarity score in [0, 1] for two strings.

    Uses rapidfuzz token_sort_ratio when available, otherwise falls
    back to a simple token-overlap Jaccard score.
    """
    if RAPIDFUZZ_AVAILABLE:
        return token_sort_ratio(a, b) / 100.0

    # Fallback: exact match
    if a == b:
        return 1.0

    # Fallback: substring containment
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return len(shorter) / len(longer) if longer else 0.0

    # Fallback: Jaccard token overlap
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ---------------------------------------------------------------------------
# SchemaAligner agent
# ---------------------------------------------------------------------------

class SchemaAligner(BaseAgent):
    """
    Merge and normalise extraction results from multiple sources.

    Accepts entities and triples from the main LLM extraction pipeline
    and from the optional Triplex extraction pipeline.  Produces a single
    deduplicated, schema-normalised set of entities and triples.

    Uses SharedMemory to:
    - Store the merged result for downstream agents

    Uses MessageBus to:
    - Optionally notify downstream agents of the merged output
    """

    # Fuzzy-match threshold for considering two entity texts as duplicates.
    ENTITY_MATCH_THRESHOLD = 0.85

    # Confidence boost applied when both sources agree on a triple.
    DUAL_SOURCE_BOOST = 0.10

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.6,
    ):
        super().__init__(
            name="SchemaAligner",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        context: AgentContext,
        main_entities: Optional[List[Dict[str, Any]]] = None,
        main_triples: Optional[List[Dict[str, Any]]] = None,
        triplex_entities: Optional[List[Dict[str, Any]]] = None,
        triplex_triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Merge and normalise extraction results from two pipelines.

        Args:
            context: Processing context (document id, domain, etc.).
            main_entities: Entities from the main LLM extraction pipeline.
            main_triples: Triples from the main LLM extraction pipeline.
            triplex_entities: Entities from the Triplex extraction pipeline.
            triplex_triples: Triples from the Triplex extraction pipeline.

        Returns:
            ExtractionResult whose ``items`` is a dict with keys
            ``"entities"`` and ``"triples"``.
        """
        self.stats["calls"] += 1

        main_entities = main_entities or []
        main_triples = main_triples or []
        triplex_entities = triplex_entities or []
        triplex_triples = triplex_triples or []

        domain = context.domain or "General"

        self.log(
            f"Aligning: {len(main_entities)} main entities + "
            f"{len(triplex_entities)} triplex entities, "
            f"{len(main_triples)} main triples + "
            f"{len(triplex_triples)} triplex triples."
        )

        # Step 1 -- normalise types/relations in both sources
        main_entities = [self._normalise_entity(e) for e in main_entities]
        triplex_entities = [self._normalise_entity(e) for e in triplex_entities]
        main_triples = [self._normalise_triple(t) for t in main_triples]
        triplex_triples = [self._normalise_triple(t) for t in triplex_triples]

        # Step 2 -- merge entities
        merged_entities = self._merge_entities(
            main_entities, triplex_entities, domain, context,
        )
        self.log(f"Merged entities: {len(merged_entities)}")

        # Step 3 -- merge triples
        merged_triples = self._merge_triples(main_triples, triplex_triples)
        self.log(f"Merged triples: {len(merged_triples)}")

        # Overall confidence: weighted average of entity and triple confidences
        entity_confs = [e.get("confidence", 0.0) for e in merged_entities]
        triple_confs = [t.get("confidence", 0.0) for t in merged_triples]
        all_confs = entity_confs + triple_confs
        avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0

        # Persist to shared memory
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "merged_entities": merged_entities,
                    "merged_triples": merged_triples,
                    "document_id": context.document_id,
                },
            )

        return ExtractionResult(
            items={"entities": merged_entities, "triples": merged_triples},
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "main_entity_count": len(main_entities),
                "triplex_entity_count": len(triplex_entities),
                "merged_entity_count": len(merged_entities),
                "main_triple_count": len(main_triples),
                "triplex_triple_count": len(triplex_triples),
                "merged_triple_count": len(merged_triples),
            },
        )

    # ------------------------------------------------------------------ #
    #  Normalisation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *entity* with its type normalised."""
        entity = dict(entity)
        raw_type = entity.get("type", "")
        if raw_type:
            entity["type"] = _to_upper_snake_case(raw_type)
        return entity

    @staticmethod
    def _normalise_triple(triple: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *triple* with its relation normalised."""
        triple = dict(triple)
        raw_relation = triple.get("relation", "")
        if raw_relation:
            triple["relation"] = _to_upper_snake_case(raw_relation)
        # Also normalise subject/object types if present
        for key in ("subject_type", "object_type"):
            raw = triple.get(key, "")
            if raw:
                triple[key] = _to_upper_snake_case(raw)
        return triple

    # ------------------------------------------------------------------ #
    #  Entity merging                                                      #
    # ------------------------------------------------------------------ #

    def _merge_entities(
        self,
        main: List[Dict[str, Any]],
        triplex: List[Dict[str, Any]],
        domain: str,
        context: AgentContext,
    ) -> List[Dict[str, Any]]:
        """Deduplicate and merge entities from both sources.

        For each Triplex entity, attempt to find a fuzzy match in the
        main set.  Matched pairs are merged (higher-confidence entry
        wins, aliases are combined).  Unmatched Triplex entities are
        appended as new discoveries.
        """
        # Work on copies so we don't mutate caller data
        merged: List[Dict[str, Any]] = [dict(e) for e in main]

        # Tag source for provenance
        for e in merged:
            e.setdefault("sources", [])
            if "main" not in e["sources"]:
                e["sources"].append("main")

        for tx_entity in triplex:
            tx_text = tx_entity.get("text", "")
            if not tx_text:
                continue

            tx_norm = _normalise_text(tx_text)
            best_idx: Optional[int] = None
            best_score = 0.0

            for idx, m_entity in enumerate(merged):
                m_norm = _normalise_text(m_entity.get("text", ""))
                if not m_norm:
                    continue
                score = _fuzzy_score(tx_norm, m_norm)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score >= self.ENTITY_MATCH_THRESHOLD:
                # --- Match found: merge ---
                merged[best_idx] = self._merge_entity_pair(
                    merged[best_idx], tx_entity, domain, context,
                )
            else:
                # --- No match: new entity from Triplex ---
                new_entity = dict(tx_entity)
                new_entity.setdefault("sources", [])
                if "triplex" not in new_entity["sources"]:
                    new_entity["sources"].append("triplex")
                merged.append(new_entity)

        return merged

    def _merge_entity_pair(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        domain: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Merge *incoming* entity into *existing*, keeping the best data.

        - The higher-confidence entry's text becomes the canonical name.
        - Aliases are merged from both sides.
        - When both sources disagree on type, the LLM is optionally
          consulted; otherwise the higher-confidence type wins.
        """
        ex_conf = existing.get("confidence", 0.0)
        in_conf = incoming.get("confidence", 0.0)

        # Pick winner for canonical text
        if in_conf > ex_conf:
            winner, loser = dict(incoming), existing
        else:
            winner, loser = dict(existing), incoming

        # Combine aliases
        aliases = set(winner.get("aliases", []))
        aliases.update(loser.get("aliases", []))
        loser_text = loser.get("text", "")
        winner_text = winner.get("text", "")
        if loser_text and loser_text != winner_text:
            aliases.add(loser_text)
        aliases.discard(winner_text)
        winner["aliases"] = sorted(aliases)

        # Boost confidence when both sources agree
        winner["confidence"] = min(1.0, max(ex_conf, in_conf) + self.DUAL_SOURCE_BOOST)

        # Track provenance
        sources = set(existing.get("sources", []))
        sources.update(incoming.get("sources", []))
        sources.add("triplex")
        winner["sources"] = sorted(sources)

        # Resolve type conflict
        ex_type = existing.get("type", "")
        in_type = incoming.get("type", "")
        if ex_type and in_type and ex_type != in_type:
            resolved_type = self._resolve_type_conflict(
                winner_text, ex_type, in_type, domain,
            )
            winner["type"] = resolved_type

        return winner

    def _resolve_type_conflict(
        self,
        entity_text: str,
        type_a: str,
        type_b: str,
        domain: str,
    ) -> str:
        """Attempt to resolve a type conflict between two sources.

        Tries to use the LLM for disambiguation.  On failure (LLM error,
        no LLM configured, etc.) falls back to the first type.
        """
        try:
            prompt = TYPE_RESOLUTION_PROMPT.format(
                entity_text=entity_text,
                type_a=type_a,
                type_b=type_b,
                domain=domain,
            )
            result = self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are a knowledge-graph schema expert. "
                    "Resolve entity-type conflicts concisely."
                ),
                tier=ModelTier.SMALL,
                temperature=0.1,
            )
            resolved = result.get("resolved_type", "")
            if resolved:
                normalised = _to_upper_snake_case(resolved)
                reason = result.get("reason", "")
                self.log(
                    f"Type conflict resolved for \"{entity_text}\": "
                    f"{type_a} vs {type_b} -> {normalised} ({reason})"
                )
                return normalised
        except Exception as exc:
            self.log(
                f"LLM type resolution failed for \"{entity_text}\" "
                f"({type_a} vs {type_b}): {exc}",
                level="WARNING",
            )

        # Fallback: keep whichever type is already UPPER_SNAKE_CASE, or type_a
        return type_a

    # ------------------------------------------------------------------ #
    #  Triple merging                                                      #
    # ------------------------------------------------------------------ #

    def _merge_triples(
        self,
        main: List[Dict[str, Any]],
        triplex: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate and merge triples from both sources.

        Triples are compared by their (subject, relation, object) tuple
        after case-insensitive normalisation.  When both sources produce
        the same triple, confidence is boosted.
        """
        # Index main triples by normalised (subject, relation, object)
        merged: List[Dict[str, Any]] = [dict(t) for t in main]
        index: Dict[Tuple[str, str, str], int] = {}

        for idx, triple in enumerate(merged):
            key = self._triple_key(triple)
            index[key] = idx
            triple.setdefault("sources", [])
            if "main" not in triple["sources"]:
                triple["sources"].append("main")

        for tx_triple in triplex:
            key = self._triple_key(tx_triple)
            if key in index:
                # Both sources agree -- boost confidence
                existing = merged[index[key]]
                ex_conf = existing.get("confidence", 0.0)
                tx_conf = tx_triple.get("confidence", 0.0)
                existing["confidence"] = min(
                    1.0, max(ex_conf, tx_conf) + self.DUAL_SOURCE_BOOST,
                )
                sources = set(existing.get("sources", []))
                sources.add("triplex")
                existing["sources"] = sorted(sources)

                # Merge evidence lists
                ex_evidence = existing.get("evidence", [])
                tx_evidence = tx_triple.get("evidence", [])
                combined = list(ex_evidence)
                for ev in tx_evidence:
                    if ev not in combined:
                        combined.append(ev)
                existing["evidence"] = combined
            else:
                # New triple from Triplex
                new_triple = dict(tx_triple)
                new_triple.setdefault("sources", [])
                if "triplex" not in new_triple["sources"]:
                    new_triple["sources"].append("triplex")
                index[key] = len(merged)
                merged.append(new_triple)

        return merged

    @staticmethod
    def _triple_key(triple: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return a case-insensitive comparison key for a triple."""
        subject = _normalise_text(triple.get("subject", ""))
        relation = _normalise_text(triple.get("relation", ""))
        obj = _normalise_text(triple.get("object", ""))
        return (subject, relation, obj)

    # ------------------------------------------------------------------ #
    #  Deliberation / voting interface                                     #
    # ------------------------------------------------------------------ #

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple["VoteType", float, str]:
        """SchemaAligner voting logic for deliberation."""
        from multi_agent_kg.core.deliberation import VoteType

        if hypothesis_type in ("entity", "triple"):
            return self._vote_on_schema_conformance(hypothesis_content, hypothesis_type)
        return VoteType.ABSTAIN, 0.5, "SchemaAligner cannot evaluate this hypothesis type"

    def _vote_on_schema_conformance(
        self,
        item: Dict[str, Any],
        item_type: str,
    ) -> Tuple["VoteType", float, str]:
        """Vote based on whether the item has well-formed schema fields."""
        from multi_agent_kg.core.deliberation import VoteType

        if item_type == "entity":
            text = item.get("text", "")
            etype = item.get("type", "")
            if not text:
                return VoteType.REJECT, 0.9, "Entity has no text"
            if not etype:
                return VoteType.WEAK_REJECT, 0.7, "Entity has no type"
            # Check type is UPPER_SNAKE_CASE
            if etype != _to_upper_snake_case(etype):
                return (
                    VoteType.WEAK_REJECT,
                    0.6,
                    f"Entity type '{etype}' is not UPPER_SNAKE_CASE",
                )
            return VoteType.WEAK_ACCEPT, 0.7, "Entity passes schema checks"

        if item_type == "triple":
            for field in ("subject", "relation", "object"):
                if not item.get(field):
                    return VoteType.REJECT, 0.9, f"Triple missing '{field}'"
            relation = item.get("relation", "")
            if relation != _to_upper_snake_case(relation):
                return (
                    VoteType.WEAK_REJECT,
                    0.6,
                    f"Relation '{relation}' is not UPPER_SNAKE_CASE",
                )
            return VoteType.WEAK_ACCEPT, 0.7, "Triple passes schema checks"

        return VoteType.ABSTAIN, 0.5, "Unknown item type"
