"""
Extraction Validator Agent (Coordinator).

Responsible for:
- Validating entity and relation extractions
- Handling escalated low-confidence items
- Running iterative refinement loops
- Coordinating blackboard voting

This is the first coordinator agent in the pipeline.
"""

from typing import Any, Dict, List, Optional
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
from multi_agent_kg.core.communication import MessageBus, CommunicationType, MessagePriority
from multi_agent_kg.core.config import LLMConfig


VALIDATION_PROMPT = """Validate these extractions for accuracy and completeness.

DOMAIN: {domain}
ORIGINAL TEXT:
{text}

ENTITIES:
{entities_json}

TRIPLES:
{triples_json}

For each item, assess:
1. Is the extraction accurate based on the text?
2. Is the confidence score appropriate?
3. Are there any errors or inconsistencies?
4. What improvements would you suggest?

Return:
{{
    "entity_validations": [
        {{
            "entity": "<entity text>",
            "valid": <true/false>,
            "adjusted_confidence": <0.0-1.0>,
            "issues": ["<issue1>", ...],
            "corrections": "<suggested correction if any>"
        }}
    ],
    "triple_validations": [
        {{
            "subject": "<subject>",
            "relation": "<relation>",
            "object": "<object>",
            "valid": <true/false>,
            "adjusted_confidence": <0.0-1.0>,
            "issues": ["<issue1>", ...],
            "corrections": "<suggested correction if any>"
        }}
    ],
    "overall_quality": <0.0-1.0>,
    "recommendations": ["<recommendation1>", ...]
}}"""


REFINEMENT_PROMPT = """Refine these extractions based on the feedback.

ORIGINAL EXTRACTIONS:
{extractions_json}

VALIDATION FEEDBACK:
{feedback_json}

REFINEMENT ITERATION: {iteration} of {max_iterations}

Apply the corrections and improve the extractions.

Return:
{{
    "refined_entities": [
        {{
            "id": "<entity id>",
            "text": "<corrected text>",
            "type": "<type>",
            "confidence": <0.0-1.0>,
            "refinement_notes": "<what was changed>"
        }}
    ],
    "refined_triples": [
        {{
            "subject": "<subject>",
            "relation": "<relation>",
            "object": "<object>",
            "confidence": <0.0-1.0>,
            "refinement_notes": "<what was changed>"
        }}
    ],
    "quality_after_refinement": <0.0-1.0>
}}"""


class ExtractionValidator(BaseAgent):
    """
    Extraction Validator Agent - Coordinates extraction validation.
    
    Responsibilities:
    1. Validate entity and relation extractions
    2. Handle escalated low-confidence items
    3. Run iterative refinement loops (max 4 iterations)
    4. Coordinate blackboard voting for ambiguous cases
    5. Send validated extractions to next stage
    
    Uses SharedMemory to:
    - Read escalated items from blackboard
    - Post validation results
    - Track refinement history
    
    Uses MessageBus to:
    - Receive escalations from worker agents
    - Request refinements from workers
    - Send validated results to VerificationAgent
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        max_iterations: int = 4,
    ):
        super().__init__(
            name="ExtractionValidator",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.LARGE,  # Coordinator uses large model
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Validate and refine extractions.
        
        Args:
            context: Processing context
            entities: Extracted entities to validate
            triples: Extracted triples to validate
            
        Returns:
            ExtractionResult with validated extractions
        """
        self.stats["calls"] += 1
        
        entities = entities or context.entities or []
        triples = triples or context.relations or []
        
        # Process any pending messages
        self._process_messages()
        
        # Process blackboard escalations
        escalations = self._process_escalations()
        
        # Add escalated items to validation queue
        for esc in escalations:
            items = esc.content.get("items", [])
            # Determine if entity or triple based on structure
            for item in items:
                if "relation" in item or "subject" in item:
                    triples.append(item)
                else:
                    entities.append(item)
        
        # Initial validation
        validation_result = self._validate_extractions(
            context.text,
            entities,
            triples,
            context.domain,
        )
        
        # Check if refinement needed
        overall_quality = validation_result.get("overall_quality", 0)
        
        current_entities = entities
        current_triples = triples
        iteration = 0
        
        # Iterative refinement loop
        while overall_quality < self.quality_threshold and iteration < self.max_iterations:
            iteration += 1
            self.log(f"Refinement iteration {iteration}/{self.max_iterations}")
            
            # Request refinement
            refined = self._refine_extractions(
                current_entities,
                current_triples,
                validation_result,
                iteration,
            )
            
            current_entities = refined.get("refined_entities", current_entities)
            current_triples = refined.get("refined_triples", current_triples)
            
            # Re-validate
            validation_result = self._validate_extractions(
                context.text,
                current_entities,
                current_triples,
                context.domain,
            )
            
            overall_quality = validation_result.get("overall_quality", 0)
            self.log(f"Quality after refinement: {overall_quality:.2f}")
        
        # Apply validation results
        validated_entities = self._apply_entity_validations(
            current_entities,
            validation_result.get("entity_validations", []),
        )
        
        validated_triples = self._apply_triple_validations(
            current_triples,
            validation_result.get("triple_validations", []),
        )
        
        # Resolve blackboard entries
        self._resolve_escalations(escalations, validated_entities, validated_triples)
        
        # Store validation results
        if self.shared_memory:
            self._store_validation_results(
                validated_entities,
                validated_triples,
                context.document_id,
                iteration,
            )
        
        # Forward to verification agent
        if self.message_bus:
            self._forward_to_verification(
                validated_entities,
                validated_triples,
                context.document_id,
            )
        
        self.log(
            f"Validated {len(validated_entities)} entities, "
            f"{len(validated_triples)} triples after {iteration} refinements"
        )
        
        return ExtractionResult(
            items={
                "entities": validated_entities,
                "triples": validated_triples,
            },
            confidence=overall_quality,
            metadata={
                "document_id": context.document_id,
                "refinement_iterations": iteration,
                "escalations_processed": len(escalations),
            },
        )

    def _process_messages(self) -> None:
        """Process incoming messages."""
        if not self.message_bus:
            return
        
        messages = self.receive_messages()
        for msg in messages:
            if msg.comm_type == CommunicationType.DELEGATE:
                # Handle delegation requests
                if msg.content.get("action") == "escalation":
                    self.log(f"Received escalation from {msg.sender}")

    def _process_escalations(self) -> List[Any]:
        """Process blackboard escalations."""
        if not self.shared_memory:
            return []
        
        return self.shared_memory.get_blackboard_entries(
            entry_type="escalation",
            status="pending",
        )

    def _validate_extractions(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        domain: Optional[str],
    ) -> Dict[str, Any]:
        """Validate extractions using LLM."""
        if not entities and not triples:
            return {"overall_quality": 0.0}
        
        entities_json = json.dumps(entities[:30], indent=2)
        triples_json = json.dumps(triples[:30], indent=2)
        
        prompt = VALIDATION_PROMPT.format(
            text=text[:4000],
            entities_json=entities_json,
            triples_json=triples_json,
            domain=domain or "general",
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert extraction validator. Be thorough but fair in assessment.",
            tier=ModelTier.LARGE,
        )
        
        return result

    def _refine_extractions(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        validation: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Refine extractions based on validation feedback."""
        extractions_json = json.dumps({
            "entities": entities[:20],
            "triples": triples[:20],
        }, indent=2)
        
        feedback_json = json.dumps({
            "entity_validations": validation.get("entity_validations", []),
            "triple_validations": validation.get("triple_validations", []),
            "recommendations": validation.get("recommendations", []),
        }, indent=2)
        
        prompt = REFINEMENT_PROMPT.format(
            extractions_json=extractions_json,
            feedback_json=feedback_json,
            iteration=iteration,
            max_iterations=self.max_iterations,
        )
        
        self.stats["refinements"] += 1
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at refining extractions. Apply corrections precisely.",
            tier=ModelTier.LARGE,
        )
        
        return result

    def _apply_entity_validations(
        self,
        entities: List[Dict[str, Any]],
        validations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply validation results to entities."""
        validation_map = {v.get("entity", ""): v for v in validations}
        
        validated = []
        for entity in entities:
            entity_text = entity.get("text", "")
            validation = validation_map.get(entity_text, {})
            
            if validation.get("valid", True):
                entity["confidence"] = validation.get("adjusted_confidence", entity.get("confidence", 0.7))
                entity["validation_issues"] = validation.get("issues", [])
                validated.append(entity)
        
        return validated

    def _apply_triple_validations(
        self,
        triples: List[Dict[str, Any]],
        validations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply validation results to triples."""
        validated = []
        
        for triple in triples:
            # Find matching validation
            matching_val = None
            for v in validations:
                if (v.get("subject") == triple.get("subject") and
                    v.get("relation") == triple.get("relation") and
                    v.get("object") == triple.get("object")):
                    matching_val = v
                    break
            
            if matching_val is None or matching_val.get("valid", True):
                if matching_val:
                    triple["confidence"] = matching_val.get("adjusted_confidence", triple.get("confidence", 0.7))
                    triple["validation_issues"] = matching_val.get("issues", [])
                validated.append(triple)
        
        return validated

    def _resolve_escalations(
        self,
        escalations: List[Any],
        validated_entities: List[Dict[str, Any]],
        validated_triples: List[Dict[str, Any]],
    ) -> None:
        """Resolve blackboard escalations."""
        if not self.shared_memory:
            return
        
        for esc in escalations:
            self.shared_memory.resolve_blackboard_entry(esc.id, "accepted")

    def _store_validation_results(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        document_id: str,
        iterations: int,
    ) -> None:
        """Store validation results in memory."""
        self.store_in_memory(
            memory_type=MemoryType.WORKING,
            content={
                "validated_entities": entities,
                "validated_triples": triples,
                "document_id": document_id,
                "refinement_iterations": iterations,
            },
        )

    def _forward_to_verification(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Forward validated extractions to verification agent."""
        self.send_message(
            receiver="ExtractionVerificationAgent",
            comm_type=CommunicationType.DELEGATE,
            content={
                "action": "verify",
                "entities": entities,
                "triples": triples,
                "document_id": document_id,
            },
            priority=MessagePriority.NORMAL,
        )
