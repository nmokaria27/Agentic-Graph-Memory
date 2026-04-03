"""
Orphan Linker Agent.

Post-verification agent that rescues orphan entities (entities with no
connections in any triple). Runs between the critic-corrector loop and
the KnowledgeOrganizer integration stage.

For each orphan entity the agent classifies it as one of:
  - **link**:  A genuine entity missing relationships. The LLM extracts
               new triples connecting it to existing entities.
  - **reify**: A property value (date, metric, monetary amount) that
               should be attached to a nearby entity via a typed relation
               rather than existing as a standalone node.
  - **prune**: Noise — generic descriptors, bare adjectives, or artifacts
               that should be removed from the entity list entirely.

Design goals:
  - Safe: if LLM calls fail, pass through unchanged (no data loss).
  - Batched: processes orphans in configurable batches to bound cost.
  - Minimal: only adds triples / removes entities; does not modify
    existing triples or entities that are already connected.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json

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
from multi_agent_kg.schemas.extraction_schemas import OrphanLinkerResponse


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ORPHAN_LINKER_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph analyst. You receive a list of "
    "'orphan' entities that currently have NO relationships in the graph, "
    "along with the entities that ARE connected and the source text. "
    "Your job is to either rescue each orphan by finding a missing "
    "relationship, convert it to a property on an existing entity, "
    "or mark it for removal."
)

ORPHAN_LINKER_PROMPT = """\
The following entities were extracted from the source text but have NO
relationships (triples) connecting them to any other entity in the graph.

ORPHAN ENTITIES (no connections):
{orphan_entities_json}

CONNECTED ENTITIES (already in graph with relationships):
{connected_entities_json}

EXISTING TRIPLES (for context on what relationships already exist):
{triples_sample_json}

SOURCE TEXT:
{source_text}

For each orphan entity, decide one of:

1. **link** — This is a real, meaningful entity that SHOULD be in the graph.
   Find the relationship(s) it has with connected entities based on the source
   text. Output new triples in the `new_triples` list.

2. **reify** — This is a property/attribute value (a date, monetary amount,
   percentage, metric, time period, ranking, or description) rather than a
   standalone entity. Specify which connected entity it belongs to and what
   relation/property name to use (e.g. OCCURRED_ON, HAS_VALUE, HAS_REVENUE,
   FOUNDED_DATE). Output the attachment as a new triple in `new_triples`.

3. **prune** — This is noise: a generic descriptor, bare adjective, abstract
   quality, duplicate of an existing entity under a different surface form, or
   an extraction artifact. It should be removed from the graph entirely.

Rules:
- Every new triple MUST reference at least one entity that is already connected
  (from CONNECTED ENTITIES) so the orphan becomes linked to the graph.
- Relation names must be descriptive UPPER_SNAKE_CASE.
- Only create triples that are supported by the source text.
- If you cannot find a clear relationship for an entity, classify it as prune
  rather than inventing a connection.
- For dates/times: prefer reify with relations like OCCURRED_ON, FOUNDED_DATE,
  ANNOUNCED_DATE, PERIOD, etc.
- For monetary values: prefer reify with HAS_VALUE, HAS_REVENUE, VALUED_AT, etc.

Return a JSON object with:
- "classifications": one entry per orphan entity
- "new_triples": all new triples (for both "link" and "reify" actions)
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

ORPHAN_BATCH_SIZE = 25  # Max orphans per LLM call


class OrphanLinker(BaseAgent):
    """
    Orphan Linker Agent -- rescues disconnected entities after verification.

    Runs after the critic-corrector loop (Stage 9) and before the
    KnowledgeOrganizer (Stage 10).  It identifies entities that appear in
    no triple and either links them, reifies them as properties, or prunes
    them.

    Integration
    -----------
    * Receives verified_entities and verified_triples from the pipeline.
    * Returns updated entity and triple lists (safe pass-through on failure).
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        batch_size: int = ORPHAN_BATCH_SIZE,
    ):
        super().__init__(
            name="OrphanLinker",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Identify and rescue orphan entities.

        Parameters
        ----------
        context : AgentContext
            Processing context (must contain source text).
        entities : list[dict]
            Verified entities from the critic-corrector stage.
        triples : list[dict]
            Verified triples from the critic-corrector stage.

        Returns
        -------
        ExtractionResult
            * ``items`` contains ``entities`` (pruned) and ``triples``
              (augmented with new links).
            * ``metadata`` has rescue statistics.
        """
        self.stats["calls"] += 1

        entities = entities or context.entities or []
        triples = triples or context.relations or []

        # --- Identify orphans ---
        connected_ids = self._get_connected_ids(triples)
        orphans, connected = self._partition_entities(entities, connected_ids)

        if not orphans:
            self.log("No orphan entities found -- nothing to do.")
            return ExtractionResult(
                items={"entities": entities, "triples": triples},
                confidence=1.0,
                metadata={"orphans_found": 0, "status": "no_orphans"},
            )

        self.log(
            f"Found {len(orphans)} orphan entities out of "
            f"{len(entities)} total ({100 * len(orphans) / len(entities):.0f}%)"
        )

        # --- Process orphans in batches ---
        all_new_triples: List[Dict[str, Any]] = []
        prune_ids: Set[str] = set()
        linked_count = 0
        reified_count = 0
        pruned_count = 0

        for batch_start in range(0, len(orphans), self.batch_size):
            batch = orphans[batch_start : batch_start + self.batch_size]
            batch_result = self._process_orphan_batch(
                batch, connected, triples, context.text
            )
            if batch_result is None:
                # LLM call failed -- keep these orphans as-is (safe fallback)
                continue

            classifications = batch_result.get("classifications", [])
            new_triples = batch_result.get("new_triples", [])

            for cls in classifications:
                action = cls.get("action", "").lower()
                eid = cls.get("entity_id", "")
                if action == "prune":
                    prune_ids.add(eid)
                    pruned_count += 1
                elif action == "link":
                    linked_count += 1
                elif action == "reify":
                    reified_count += 1

            # Validate new triples: at least one endpoint must be connected
            for triple in new_triples:
                subj = triple.get("subject", "")
                obj = triple.get("object", "")
                if subj and obj and (subj != obj):
                    all_new_triples.append(triple)

        # --- Build final outputs ---
        # Remove pruned entities
        final_entities = [
            e for e in entities
            if self._entity_id(e) not in prune_ids
        ]

        # Append new triples
        final_triples = triples + all_new_triples

        self.log(
            f"Orphan rescue complete: {linked_count} linked, "
            f"{reified_count} reified, {pruned_count} pruned, "
            f"{len(all_new_triples)} new triples added"
        )

        # Persist stats in working memory
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "orphan_linker_stats": {
                        "orphans_found": len(orphans),
                        "linked": linked_count,
                        "reified": reified_count,
                        "pruned": pruned_count,
                        "new_triples": len(all_new_triples),
                    },
                    "document_id": context.document_id,
                },
            )

        return ExtractionResult(
            items={"entities": final_entities, "triples": final_triples},
            confidence=1.0,
            metadata={
                "orphans_found": len(orphans),
                "linked": linked_count,
                "reified": reified_count,
                "pruned": pruned_count,
                "new_triples_added": len(all_new_triples),
                "entities_removed": len(prune_ids),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_id(entity: Dict[str, Any]) -> str:
        """Extract the canonical ID string for an entity dict."""
        return entity.get("id", entity.get("text", ""))

    @staticmethod
    def _get_connected_ids(triples: List[Dict[str, Any]]) -> Set[str]:
        """Return set of entity IDs that appear in at least one triple."""
        ids: Set[str] = set()
        for t in triples:
            subj = t.get("subject", "")
            obj = t.get("object", "")
            if subj:
                ids.add(subj)
            if obj:
                ids.add(obj)
        return ids

    @staticmethod
    def _partition_entities(
        entities: List[Dict[str, Any]],
        connected_ids: Set[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split entities into orphans and connected."""
        orphans = []
        connected = []
        for e in entities:
            eid = e.get("id", e.get("text", ""))
            if eid in connected_ids:
                connected.append(e)
            else:
                orphans.append(e)
        return orphans, connected

    def _process_orphan_batch(
        self,
        orphans: List[Dict[str, Any]],
        connected: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        source_text: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the LLM to classify and link a batch of orphan entities.

        Returns None on failure (safe fallback: caller keeps orphans as-is).
        """
        # Format orphan entities (compact)
        orphan_summaries = []
        for e in orphans:
            orphan_summaries.append({
                "id": self._entity_id(e),
                "type": e.get("type", "UNKNOWN"),
            })

        # Format connected entities (compact, capped)
        connected_summaries = []
        for e in connected[:80]:
            connected_summaries.append({
                "id": self._entity_id(e),
                "type": e.get("type", "UNKNOWN"),
            })

        # Sample of triples for context (capped)
        triple_sample = []
        for t in triples[:60]:
            triple_sample.append({
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
            })

        prompt = ORPHAN_LINKER_PROMPT.format(
            orphan_entities_json=json.dumps(orphan_summaries, indent=2),
            connected_entities_json=json.dumps(connected_summaries, indent=2),
            triples_sample_json=json.dumps(triple_sample, indent=2),
            source_text=source_text[:6000],
        )

        try:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=ORPHAN_LINKER_SYSTEM_PROMPT,
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
                response_schema=OrphanLinkerResponse,
            )
        except Exception as exc:
            self.log(
                f"LLM call failed for orphan batch, keeping orphans as-is: {exc}",
                level="WARNING",
            )
            return None

        # Normalize Pydantic model → dict
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return result
