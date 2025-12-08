"""
Entity Extractor Agent.

Implements multi-stage entity extraction pipeline:
1. Initial Extraction: Identify candidate entities
2. Boundary Refinement: Fix entity boundaries
3. Type Assignment: Classify entity types
4. Coreference Resolution: Link mentions to same entity

Features:
- Self-consistency for confidence estimation
- Cross-document entity resolution via SharedMemory
- Blackboard posting for ambiguous entities
- Iterative refinement with feedback
"""

from typing import Any, Dict, List, Optional, Tuple
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
    

INITIAL_EXTRACTION_PROMPT = """Extract all named entities from the following text.
Focus on: {entity_types}

TEXT:
{text}

Return a JSON object with:
{{
    "entities": [
        {{
            "text": "<exact entity mention>",
            "start": <character start position>,
            "end": <character end position>,
            "type_guess": "<entity type guess>"
        }}
    ]
}}

Extract ALL entities mentioned, even if you're uncertain about the type."""


BOUNDARY_REFINEMENT_PROMPT = """Review these entity extractions and fix any boundary errors.

TEXT: {text}

ENTITIES:
{entities_json}

For each entity, verify:
1. The text is complete (not cut off)
2. No extra words included
3. Start/end positions are correct

Return corrected entities:
{{
    "entities": [
        {{
            "text": "<corrected text>",
            "original_text": "<original text>",
            "start": <corrected start>,
            "end": <corrected end>,
            "boundary_fixed": <true/false>
        }}
    ]
}}"""


TYPE_ASSIGNMENT_PROMPT = """Assign entity types to these entities based on context.

DOMAIN: {domain}
VALID TYPES: {valid_types}

TEXT CONTEXT:
{text}

ENTITIES:
{entities_json}

For each entity, determine the most appropriate type from the valid types.

Return:
{{
    "entities": [
        {{
            "text": "<entity text>",
            "type": "<assigned type from valid types>",
            "type_confidence": <0.0-1.0>,
            "type_reasoning": "<brief reasoning>"
        }}
    ]
}}"""


COREFERENCE_PROMPT = """Identify which entity mentions refer to the same real-world entity.

TEXT:
{text}

ENTITIES:
{entities_json}

KNOWN ENTITIES FROM PREVIOUS DOCUMENTS:
{known_entities}

Group entities that refer to the same thing. Assign a canonical ID to each group.

Return:
{{
    "entity_groups": [
        {{
            "canonical_id": "<unique identifier>",
            "canonical_name": "<primary name>",
            "type": "<entity type>",
            "mentions": ["<mention1>", "<mention2>", ...],
            "is_known_entity": <true if matches known entity, false otherwise>
        }}
    ]
}}"""


class EntityExtractor(BaseAgent):
    """
    Entity Extractor Agent - Multi-stage entity extraction with confidence.
    
    Pipeline:
    1. Initial Extraction: Identify candidate entities with LLM
    2. Boundary Refinement: Fix entity boundaries
    3. Type Assignment: Classify entity types based on domain
    4. Coreference Resolution: Link mentions to same entity
    
    Uses SharedMemory to:
    - Get known entities for coreference
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

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract entities using multi-stage pipeline.
        
        Args:
            context: Processing context
            segments: Document segments
            domain_config: Domain configuration with entity types
            
        Returns:
            ExtractionResult with extracted entities
        """
        self.stats["calls"] += 1
        
        # Get entity types from domain config or use general
        entity_types = domain_config.get("entity_types", []) if domain_config else []
        if not entity_types:
            entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT"]
        
        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "entity_types" in msg.content:
                    entity_types = msg.content["entity_types"]
        
        # Process each segment
        all_entities = []
        low_confidence_entities = []
        
        texts_to_process = []
        if segments:
            texts_to_process = [(s.get("text", ""), s.get("segment_id")) for s in segments]
        elif context.text:
            texts_to_process = [(context.text, f"{context.document_id}_full")]
        
        for text, segment_id in texts_to_process:
            if not text:
                continue
            
            # Stage 1: Initial Extraction
            candidates = self._stage1_initial_extraction(text, entity_types)
            
            # Stage 2: Boundary Refinement
            refined = self._stage2_boundary_refinement(text, candidates)
            
            # Stage 3: Type Assignment
            typed = self._stage3_type_assignment(
                text, 
                refined, 
                entity_types,
                context.domain,
            )
            
            # Separate high and low confidence
            for entity in typed:
                entity["source_segment"] = segment_id
                if entity.get("confidence", 0) >= self.quality_threshold:
                    all_entities.append(entity)
                else:
                    low_confidence_entities.append(entity)
        
        # Stage 4: Coreference Resolution (across all segments)
        known_entities = self._get_known_entities()
        resolved = self._stage4_coreference_resolution(
            context.text,
            all_entities,
            known_entities,
        )
        
        # Handle low confidence entities
        if low_confidence_entities:
            self._handle_low_confidence_entities(
                low_confidence_entities,
                context,
            )
        
        # Store results
        if self.shared_memory:
            self._store_entities(resolved, context.document_id)
        
        # Calculate overall confidence
        if resolved:
            avg_confidence = sum(e.get("confidence", 0.5) for e in resolved) / len(resolved)
        else:
            avg_confidence = 0.0
        
        self.log(f"Extracted {len(resolved)} entities (avg confidence: {avg_confidence:.2f})")
        
        return ExtractionResult(
            items=resolved,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_entities),
                "stages_completed": 4,
            },
            needs_escalation=len(low_confidence_entities) > 0,
            escalation_reason=f"{len(low_confidence_entities)} low confidence entities" if low_confidence_entities else None,
        )

    def _stage1_initial_extraction(
        self,
        text: str,
        entity_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Stage 1: Initial entity extraction."""
        prompt = INITIAL_EXTRACTION_PROMPT.format(
            text=text,
            entity_types=", ".join(entity_types),
        )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert entity extractor. Extract all named entities precisely.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert entity extractor. Extract all named entities precisely.",
                tier=ModelTier.MEDIUM,
            )
            confidence = 0.7
        
        entities = result.get("entities", [])
        for e in entities:
            e["stage1_confidence"] = confidence
        
        return entities

    def _stage2_boundary_refinement(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Refine entity boundaries."""
        if not entities:
            return []
        
        import json
        entities_json = json.dumps(entities, indent=2)
        
        prompt = BOUNDARY_REFINEMENT_PROMPT.format(
            text=text,
            entities_json=entities_json,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at identifying precise entity boundaries.",
            tier=ModelTier.SMALL,  # Simpler task
        )
        
        return result.get("entities", entities)

    def _stage3_type_assignment(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        valid_types: List[str],
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Stage 3: Assign entity types."""
        if not entities:
            return []
        
        import json
        entities_json = json.dumps(entities, indent=2)
        
        prompt = TYPE_ASSIGNMENT_PROMPT.format(
            text=text,
            entities_json=entities_json,
            valid_types=", ".join(valid_types),
            domain=domain or "general",
        )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert at entity typing. Assign accurate types.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at entity typing. Assign accurate types.",
                tier=ModelTier.MEDIUM,
            )
            confidence = 0.7
        
        typed_entities = result.get("entities", entities)
        
        # Combine stage confidences
        for e in typed_entities:
            type_conf = e.get("type_confidence", 0.7)
            stage1_conf = e.get("stage1_confidence", 0.7)
            # Combined confidence
            e["confidence"] = (stage1_conf + type_conf + confidence) / 3
        
        return typed_entities

    def _stage4_coreference_resolution(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        known_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 4: Coreference resolution."""
        if not entities:
            return []
        
        import json
        entities_json = json.dumps(entities, indent=2)
        known_json = json.dumps(known_entities[:20], indent=2) if known_entities else "[]"
        
        prompt = COREFERENCE_PROMPT.format(
            text=text[:3000],  # Limit context
            entities_json=entities_json,
            known_entities=known_json,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at coreference resolution. Group mentions accurately.",
            tier=ModelTier.MEDIUM,
        )
        
        # Convert groups back to entity format
        resolved = []
        for group in result.get("entity_groups", []):
            resolved.append({
                "id": group.get("canonical_id", ""),
                "text": group.get("canonical_name", ""),
                "type": group.get("type", "UNKNOWN"),
                "mentions": group.get("mentions", []),
                "confidence": 0.8 if group.get("is_known_entity") else 0.7,
                "is_known_entity": group.get("is_known_entity", False),
            })
        
        # Register aliases in shared memory
        if self.shared_memory:
            for entity in resolved:
                canonical_id = entity["id"]
                for mention in entity.get("mentions", []):
                    if mention != entity["text"]:
                        self.shared_memory.register_entity_alias(mention, canonical_id)
        
        return resolved if resolved else entities

    def _get_known_entities(self) -> List[Dict[str, Any]]:
        """Get known entities from memory and knowledge graph."""
        known = []
        
        # From knowledge graph
        if self.knowledge_graph:
            for entity_id, entity in list(self.knowledge_graph.entities.items())[:50]:
                known.append({
                    "id": entity_id,
                    "text": entity.labels[0] if entity.labels else entity_id,
                    "type": entity.type,
                })
        
        # From shared memory
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=20)
            for mem in memories:
                if "entities" in mem.content:
                    known.extend(mem.content["entities"][:10])
        
        return known

    def _handle_low_confidence_entities(
        self,
        entities: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle low confidence entities via escalation."""
        # Post to blackboard for voting
        for entity in entities[:10]:  # Limit
            self.post_hypothesis(
                hypothesis=entity,
                confidence=entity.get("confidence", 0.5),
                evidence=[entity.get("source_segment", "")],
            )
        
        # Escalate to coordinator
        self.escalate_to_coordinator(
            reason="Low confidence entity extractions",
            items=entities,
            context={
                "document_id": context.document_id,
                "domain": context.domain,
            },
        )

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
