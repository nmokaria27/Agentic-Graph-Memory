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
    

INITIAL_EXTRACTION_PROMPT = """Extract ALL significant entities that represent knowledge in this document.

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
- Important statistical markers (e.g., "HbA1c", "IESS", "CFR") when they represent specific measurements

DO NOT EXTRACT:
- Generic dates/times ("January 2020", "90 days") unless defining eras/periods
- Bare numbers without meaning ("0.85", "38614")
- Common adjectives alone ("high", "low", "greater")
- Generic temporal references ("baseline", "follow-up") unless technical terms

IMPORTANT: Err on the side of INCLUSION. Extract entities that help build a comprehensive knowledge representation of the document.

{entity_guidance}

TEXT:
{text}

Return a JSON object with:
{{
    "entities": [
        {{
            "text": "<exact entity mention>",
            "start": <character start position>,
            "end": <character end position>,
            "type_guess": "<describe what this entity represents in this context>"
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


TYPE_ASSIGNMENT_PROMPT = """Assign entity types to these entities based on what they represent IN THIS DOCUMENT.

IMPORTANT:
- DISCOVER types from the content - do NOT use standard taxonomies
- Create specific, descriptive type names based on what the entity actually IS
- Types should be in UPPER_SNAKE_CASE
- Be as specific as possible (e.g., CLINICAL_MEASUREMENT not MEASUREMENT)

DOMAIN: {domain}
{type_guidance}

TEXT CONTEXT:
{text}

ENTITIES:
{entities_json}

For each entity, assign a type that describes what it represents in this specific document.

Return:
{{
    "entities": [
        {{
            "text": "<entity text>",
            "type": "<DISCOVERED_TYPE_NAME>",
            "type_confidence": <0.0-1.0>,
            "type_reasoning": "<why this type fits this entity>"
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
        entity_types_raw = domain_config.get("entity_types", []) if domain_config else []
        entity_types = self._normalize_entity_types(entity_types_raw)
        
        if not entity_types:
            entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT"]
        
        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "entity_types" in msg.content:
                    entity_types = self._normalize_entity_types(msg.content["entity_types"])
        
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
        print(f"\n[ENTITY EXTRACTOR DEBUG]")
        print(f"  Total extracted: {len(all_entities)}")
        print(f"  High confidence (>={self.quality_threshold}): {len(all_entities)}")
        print(f"  Low confidence (<{self.quality_threshold}): {len(low_confidence_entities)}")
        
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
        # Build guidance from entity types if provided by domain classifier
        entity_guidance = ""
        if entity_types:
            entity_types_str = self._normalize_entity_types(entity_types)
            entity_guidance = f"The domain classifier suggested these entity categories for context: {', '.join(entity_types_str)}\nHowever, feel free to discover additional entity types based on the actual content."
        
        prompt = INITIAL_EXTRACTION_PROMPT.format(
            text=text,
            entity_guidance=entity_guidance,
        )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert at discovering entities from scratch. Extract entities based on what you observe in the text, not predefined categories.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at discovering entities from scratch. Extract entities based on what you observe in the text, not predefined categories.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
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
        
        # Process in batches to avoid JSON truncation
        batch_size = 30
        refined_entities = []
        
        import json
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            entities_json = json.dumps(batch, indent=2)
            
            prompt = BOUNDARY_REFINEMENT_PROMPT.format(
                text=text[:3000],  # Limit text size
                entities_json=entities_json,
            )
            
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at identifying precise entity boundaries.",
                tier=ModelTier.SMALL,  # Simpler task
                max_tokens=4096,
            )
            
            refined_entities.extend(result.get("entities", batch))
        
        return refined_entities

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
        
        # Process in batches to avoid token limit issues
        batch_size = 25  # Process 25 entities at a time to stay under token limit
        all_typed_entities = []
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            entities_json = json.dumps(batch, indent=2)
            
            # Build type guidance
            type_guidance = ""
            if valid_types:
                type_guidance = f"Suggested type categories from domain analysis: {', '.join(valid_types)}\\nYou may use these or create more specific types as needed."
            
            prompt = TYPE_ASSIGNMENT_PROMPT.format(
                text=text,
                entities_json=entities_json,
                type_guidance=type_guidance,
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
                    max_tokens=4096,
                )
                confidence = 0.7
            
            typed_batch = result.get("entities", batch)
            
            # Combine stage confidences
            for e in typed_batch:
                type_conf = e.get("type_confidence", 0.7)
                stage1_conf = e.get("stage1_confidence", 0.7)
                # Combined confidence
                e["confidence"] = (stage1_conf + type_conf + confidence) / 3
            
            all_typed_entities.extend(typed_batch)
        
        return all_typed_entities

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
        
        # Process in batches to avoid token limit
        batch_size = 20  # Smaller batches for coreference resolution
        all_resolved = []
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            entities_json = json.dumps(batch, indent=2)
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
                max_tokens=4096,
            )
            
            # Convert groups back to entity format
            for group in result.get("entity_groups", []):
                resolved_entity = {
                    "id": group.get("canonical_id", ""),
                    "text": group.get("canonical_name", ""),
                    "type": group.get("type", "UNKNOWN"),
                    "mentions": group.get("mentions", []),
                    "confidence": 0.8 if group.get("is_known_entity") else 0.7,
                    "is_known_entity": group.get("is_known_entity", False),
                }
                all_resolved.append(resolved_entity)
                
                # Register aliases in shared memory
                if self.shared_memory:
                    canonical_id = resolved_entity["id"]
                    for mention in resolved_entity.get("mentions", []):
                        if mention != resolved_entity["text"]:
                            self.shared_memory.register_entity_alias(mention, canonical_id)
        
        return all_resolved if all_resolved else entities

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
        
        # Basic validation checks
        if not entity_text or len(entity_text) < 2:
            return VoteType.REJECT, 0.9, "Entity text too short or empty"
        
        if len(entity_text) > 100:
            return VoteType.WEAK_REJECT, 0.7, "Entity text suspiciously long"
        
        # Check if entity type is valid
        valid_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", 
                       "PRODUCT", "DISEASE", "DRUG", "GENE", "LAW", "COURT"]
        if entity_type and entity_type.upper() not in valid_types:
            return VoteType.WEAK_REJECT, 0.6, f"Unknown entity type: {entity_type}"
        
        # Check if it looks like a real entity (capitalized, etc.)
        if entity_text[0].isupper():
            return VoteType.WEAK_ACCEPT, 0.7, "Entity appears to be properly capitalized"
        
        # Default to weak accept if nothing wrong
        return VoteType.WEAK_ACCEPT, 0.6, "Entity passes basic validation"

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
