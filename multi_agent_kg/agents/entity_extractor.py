"""
Entity Extractor Agent.

Implements a consolidated entity extraction pipeline (reduced from 4 to 2 stages):
1. Combined Extraction + Type Assignment: Extract entities with types in one pass
2. Gleaning: Re-run extraction to catch missed entities (GraphRAG-style)

Coreference resolution has been moved to a separate EntityResolver agent
that operates globally across all segments (KGGen-style clustering).

Features:
- Consolidated prompts to reduce error compounding (was 4 LLM calls, now 2)
- GraphRAG-style gleaning to recover missed entities
- Constrained decoding via Pydantic schemas (eliminates JSON failures)
- Self-consistency for confidence estimation
- Accepts prior extractions from FastExtractor (GLiNER) for refinement
- Blackboard posting for ambiguous entities
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType


@dataclass
class EntityCandidate:
    """Candidate entity during extraction pipeline."""
    text: str
    start: int = 0
    end: int = 0
    entity_type: Optional[str] = None
    confidence: float = 0.5
    source_segment: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    

CONSOLIDATED_EXTRACTION_PROMPT = """Think step by step: first identify all key entities in this text, then classify each by type.

Extract ALL significant entities from the text below. For each entity, provide its exact
mention text and assign a specific type in UPPER_SNAKE_CASE.

EXTRACT entities including:
- Domain concepts, theories, methods, techniques, phenomena, mechanisms
- Medical/scientific conditions, diseases, treatments, clinical measures, biomarkers
- Therapeutic interventions, drugs, procedures, therapies
- Biological processes, pathways, molecular mechanisms
- Clinical outcomes, complications, risk factors, prognostic indicators
- Organizations, institutions, research groups, medical centers
- Researchers, authors, key contributors
- Specialized terminology and technical concepts
- Study cohorts, patient populations, demographic groups
- Measurement tools, instruments, assessment methods, scoring systems

DO NOT EXTRACT:
- Generic dates/times unless defining eras/periods
- Bare numbers without meaning
- Common adjectives alone
- Generic temporal references unless they are technical terms

IMPORTANT DISTINCTION - entities vs facts:
- An entity is a NAMED THING: a person, company, product, place, organization, or specific concept with a proper name.
- Do NOT extract rankings, descriptions, or metrics as entities (e.g., "largest company by market capitalization" is a FACT about an entity, not an entity itself).
- Do NOT extract financial figures as entities (e.g., "valued at $4 trillion" is a property, not an entity).
- Do NOT extract abstract qualities as entities (e.g., "customer loyalty", "market share" are attributes, not entities).
- Instead, these should become part of RELATIONSHIPS in a later stage.

IMPORTANT: Err on the side of INCLUSION for genuine named entities. Assign specific, descriptive types
based on what the entity actually represents in THIS context (not generic labels).

{entity_guidance}

{prior_entities_section}

TEXT:
{text}

Return a JSON object with an "entities" list."""


GLEANING_PROMPT = """The following entities were already extracted from this text.
Review the text carefully and find any entities that were MISSED.

Focus on:
- Implicit entities mentioned indirectly
- Abbreviations and acronyms
- Entities mentioned only in passing
- Entities that are part of compound noun phrases
- Entities referenced by pronouns or shortened forms
- Named locations (cities, regions, headquarters)
- Named companies mentioned as competitors or partners
- Specific product names (including older/historical products)

ALREADY FOUND ENTITIES:
{found_entities}

TEXT:
{text}

Return ONLY the NEW entities that were missed. Do not repeat already-found entities.
Return a JSON object with an "entities" list. If no new entities are found, return {{"entities": []}}."""


class EntityExtractor(BaseAgent):
    """
    Entity Extractor Agent - Consolidated extraction with gleaning.

    Pipeline (reduced from 4 stages to 2):
    1. Combined Extraction + Type Assignment in a single prompt
    2. Gleaning pass to recover missed entities (GraphRAG-style)

    Coreference resolution is handled by the separate EntityResolver agent.

    Uses SharedMemory to:
    - Store entity aliases for cross-document resolution
    - Post ambiguous entities to blackboard

    Uses MessageBus to:
    - Receive domain info from DomainClassifier
    - Send entities to RelationExtractor
    - Escalate low-confidence entities
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        use_self_consistency: bool = True,
        n_consistency_samples: int = 3,
        max_gleanings: int = 1,
    ):
        super().__init__(
            name="EntityExtractor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.use_self_consistency = use_self_consistency
        self.n_consistency_samples = n_consistency_samples
        self.max_gleanings = max_gleanings

    def _normalize_entity_types(self, entity_types_raw: Any) -> List[str]:
        """Normalize entity types to list of strings, handling dict format from DomainClassifier."""
        if not entity_types_raw:
            return []
        
        if not isinstance(entity_types_raw, list):
            return []
        
        normalized = []
        for et in entity_types_raw:
            if isinstance(et, dict):
                # DomainClassifier format: {"type": "PERSON", "description": "...", "priority": "high"}
                normalized.append(et.get("type", str(et)))
            elif isinstance(et, str):
                normalized.append(et)
            else:
                normalized.append(str(et))
        
        return normalized

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        prior_entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract entities using consolidated pipeline with gleaning.

        Args:
            context: Processing context
            segments: Document segments
            domain_config: Domain configuration with entity types
            prior_entities: Baseline entities from FastExtractor (GLiNER)
                to refine rather than extract from scratch

        Returns:
            ExtractionResult with extracted entities
        """
        self.stats["calls"] += 1

        # Get entity types from domain config or use general
        entity_types_raw = domain_config.get("entity_types", []) if domain_config else []
        entity_types = self._normalize_entity_types(entity_types_raw)

        # If no entity types from domain classifier, let the LLM discover them
        # (no hardcoded fallback — the system learns everything)

        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "entity_types" in msg.content:
                    entity_types = self._normalize_entity_types(msg.content["entity_types"])

        # Batch segments to reduce LLM calls (5 segments per batch)
        BATCH_SIZE = 5
        all_entities: List[Dict[str, Any]] = []
        low_confidence_entities: List[Dict[str, Any]] = []

        raw_segments = []
        if segments:
            raw_segments = [(s.get("text", ""), s.get("segment_id", "")) for s in segments]
        elif context.text:
            raw_segments = [(context.text, f"{context.document_id}_full")]

        # Group segments into batches
        batches = []
        for i in range(0, len(raw_segments), BATCH_SIZE):
            batches.append(raw_segments[i:i + BATCH_SIZE])

        print(f"  Processing {len(raw_segments)} segments in {len(batches)} batches")

        for batch_idx, batch in enumerate(batches):
            # Combine batch texts with separators
            combined_text = "\n\n---\n\n".join(text for text, _ in batch if text)
            batch_ids = [sid for _, sid in batch]

            if not combined_text:
                continue

            # Collect prior entities for this batch
            batch_priors = []
            if prior_entities:
                batch_priors = [
                    e for e in prior_entities
                    if e.get("source_segment") in batch_ids
                    or not e.get("source_segment")
                ]
                # Deduplicate by text
                seen = set()
                deduped = []
                for e in batch_priors:
                    key = e.get("text", "").lower()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(e)
                batch_priors = deduped[:40]  # Cap to avoid prompt overflow

            # Stage 1: Consolidated Extraction + Type Assignment (one call per batch)
            extracted = self._stage1_extract_with_types(
                combined_text, entity_types, context.domain, batch_priors,
            )

            # Stage 2: Gleaning — one pass per batch
            for _ in range(self.max_gleanings):
                gleaned = self._stage2_gleaning(combined_text, extracted)
                if not gleaned:
                    break
                extracted.extend(gleaned)

            # Tag entities with batch info
            batch_label = batch_ids[0] if batch_ids else f"batch_{batch_idx}"
            for entity in extracted:
                if not entity.get("source_segment"):
                    entity["source_segment"] = batch_label
                if entity.get("confidence", 0) >= self.quality_threshold:
                    all_entities.append(entity)
                else:
                    low_confidence_entities.append(entity)

            print(f"    Batch {batch_idx + 1}/{len(batches)}: {len(extracted)} entities")

        # Handle low confidence entities
        print(f"\n[ENTITY EXTRACTOR DEBUG]")
        print(f"  Total extracted: {len(all_entities) + len(low_confidence_entities)}")
        print(f"  High confidence (>={self.quality_threshold}): {len(all_entities)}")
        print(f"  Low confidence (<{self.quality_threshold}): {len(low_confidence_entities)}")

        if low_confidence_entities:
            self._handle_low_confidence_entities(low_confidence_entities, context)

        # Store results
        if self.shared_memory:
            self._store_entities(all_entities, context.document_id)

        # Calculate overall confidence
        if all_entities:
            avg_confidence = sum(e.get("confidence", 0.5) for e in all_entities) / len(all_entities)
        else:
            avg_confidence = 0.0

        self.log(f"Extracted {len(all_entities)} entities (avg confidence: {avg_confidence:.2f})")

        return ExtractionResult(
            items=all_entities,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_entities),
                "stages_completed": 2,
                "prior_entities_used": len(prior_entities) if prior_entities else 0,
            },
            needs_escalation=len(low_confidence_entities) > 0,
            escalation_reason=(
                f"{len(low_confidence_entities)} low confidence entities"
                if low_confidence_entities else None
            ),
        )

    def _stage1_extract_with_types(
        self,
        text: str,
        entity_types: List[str],
        domain: Optional[str],
        prior_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Stage 1: Combined entity extraction + type assignment in one prompt."""
        from multi_agent_kg.schemas.extraction_schemas import EntityExtractionResponse

        # Build guidance from entity types
        entity_guidance = ""
        if entity_types:
            entity_types_str = self._normalize_entity_types(entity_types)
            entity_guidance = (
                f"DOMAIN: {domain or 'general'}\n"
                f"Suggested entity categories: {', '.join(entity_types_str)}\n"
                f"You may also discover additional types based on the content."
            )

        # Build prior entities section if GLiNER provided a baseline
        prior_section = ""
        if prior_entities:
            prior_list = ", ".join(
                f'"{e.get("text", "")}" ({e.get("type", "?")})'
                for e in prior_entities[:30]
            )
            prior_section = (
                f"A preliminary scan found these entities: {prior_list}\n"
                f"Review them for accuracy, correct any errors, add missed entities, "
                f"and assign proper types."
            )

        prompt = CONSOLIDATED_EXTRACTION_PROMPT.format(
            text=text,
            entity_guidance=entity_guidance,
            prior_entities_section=prior_section,
        )

        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at discovering entities and classifying them. "
                    "Extract entities and assign specific UPPER_SNAKE_CASE types."
                ),
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at discovering entities and classifying them. "
                    "Extract entities and assign specific UPPER_SNAKE_CASE types."
                ),
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
                response_schema=EntityExtractionResponse,
            )
            confidence = 0.7

        entities = self._normalize_entity_list(result.get("entities", []))
        for e in entities:
            # Use LLM-provided confidence if available, otherwise use consistency score
            if "confidence" not in e or e["confidence"] == 0.7:
                e["confidence"] = confidence
        return entities

    def _stage2_gleaning(
        self,
        text: str,
        found_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Gleaning — find entities missed in the first pass."""
        from multi_agent_kg.schemas.extraction_schemas import EntityExtractionResponse

        if not found_entities:
            return []

        found_list = ", ".join(
            f'"{e.get("text", "") if isinstance(e, dict) else str(e)}"'
            for e in found_entities[:50]
        )

        prompt = GLEANING_PROMPT.format(
            text=text,
            found_entities=found_list,
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at finding missed entities. Only return NEW entities.",
            tier=ModelTier.MEDIUM,
            max_tokens=2048,
            response_schema=EntityExtractionResponse,
        )

        new_entities = self._normalize_entity_list(result.get("entities", []))
        # Lower confidence for gleaned entities (they were missed initially)
        for e in new_entities:
            e["confidence"] = min(e.get("confidence", 0.6), 0.7)
            e["gleaned"] = True
        return new_entities

    @staticmethod
    def _normalize_entity_list(raw: list) -> List[Dict[str, Any]]:
        """Normalise a list of entities that may be dicts, strings, or have
        non-standard keys (e.g. 'mention' instead of 'text')."""
        normalised = []
        for item in raw:
            if isinstance(item, str):
                # LLM returned a bare string — wrap it
                normalised.append({"text": item, "type": "UNKNOWN", "confidence": 0.5})
            elif isinstance(item, dict):
                # Accept 'mention' or 'name' as aliases for 'text'
                if "text" not in item:
                    item["text"] = item.pop("mention", item.pop("name", item.pop("entity", "")))
                if "type" not in item:
                    item["type"] = item.pop("entity_type", item.pop("type_guess", "UNKNOWN"))
                normalised.append(item)
            # Skip anything else (None, int, …)
        return normalised

    def _handle_low_confidence_entities(
        self,
        entities: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle low confidence entities via deliberation."""
        # Submit to deliberation for multi-agent voting
        for entity in entities[:10]:  # Limit to avoid flooding
            self.submit_for_deliberation(
                hypothesis_type="entity",
                content=entity,
                confidence=entity.get("confidence", 0.5),
                evidence=[entity.get("source_segment", "")],
                document_id=context.document_id,
            )
        
        # Also escalate to coordinator for awareness
        self.escalate_to_coordinator(
            reason="Low confidence entity extractions submitted for deliberation",
            items=entities,
            context={
                "document_id": context.document_id,
                "domain": context.domain,
            },
        )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        EntityExtractor's logic for voting on hypotheses.
        
        Can vote on:
        - entity: Check if it looks like a valid entity
        - relation: Check if entities exist
        - triple: Check if entities are valid
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        if hypothesis_type == "entity":
            # Evaluate entity hypothesis
            return self._vote_on_entity(hypothesis_content, context)
        elif hypothesis_type == "relation":
            # Check if the entities in the relation exist
            return self._vote_on_relation_entities(hypothesis_content, context)
        elif hypothesis_type == "triple":
            # Check if subject/object are valid entities
            return self._vote_on_triple_entities(hypothesis_content, context)
        
        return VoteType.ABSTAIN, 0.5, "EntityExtractor cannot evaluate this hypothesis type"

    def _vote_on_entity(
        self,
        entity: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on an entity hypothesis."""
        from multi_agent_kg.core.deliberation import VoteType
        
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "")

        # Basic validation — no hardcoded type lists; the system discovers types
        if not entity_text or len(entity_text) < 2:
            return VoteType.REJECT, 0.9, "Entity text too short or empty"

        if len(entity_text) > 100:
            return VoteType.WEAK_REJECT, 0.7, "Entity text suspiciously long"

        if not entity_type or entity_type == "UNKNOWN":
            return VoteType.WEAK_REJECT, 0.6, "Entity has no assigned type"

        # Accept any entity that has a non-empty type (types are domain-discovered)
        return VoteType.WEAK_ACCEPT, 0.65, f"Entity has discovered type: {entity_type}"

    def _vote_on_relation_entities(
        self,
        relation: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on whether a relation's entities are valid."""
        from multi_agent_kg.core.deliberation import VoteType
        
        subject = relation.get("subject", {})
        obj = relation.get("object", {})
        
        # Check if entities look valid
        subject_text = subject.get("text", "") if isinstance(subject, dict) else str(subject)
        obj_text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
        
        if not subject_text or not obj_text:
            return VoteType.REJECT, 0.9, "Missing subject or object entity"
        
        return VoteType.WEAK_ACCEPT, 0.6, "Entities in relation appear valid"

    def _vote_on_triple_entities(
        self,
        triple: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on whether a triple's entities are valid."""
        from multi_agent_kg.core.deliberation import VoteType
        
        subject = triple.get("subject", "")
        obj = triple.get("object", "")
        
        if not subject or not obj:
            return VoteType.REJECT, 0.9, "Triple missing subject or object"
        
        return VoteType.WEAK_ACCEPT, 0.6, "Triple entities appear valid"

    def _store_entities(
        self,
        entities: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store extracted entities in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "entities": entities,
                "document_id": document_id,
            },
        )
        
        # Add context for each entity
        for entity in entities:
            self.shared_memory.add_entity_context(
                entity_id=entity.get("id", entity.get("text", "")),
                context={
                    "document_id": document_id,
                    "type": entity.get("type"),
                    "confidence": entity.get("confidence"),
                },
            )
