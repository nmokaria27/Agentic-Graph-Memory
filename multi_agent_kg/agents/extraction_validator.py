"""
Extraction Validator Agent (Coordinator).

Responsible for:
- Validating entity and relation extractions
- Handling escalated low-confidence items
- Running iterative refinement loops
- Coordinating multi-agent deliberation and debate
- Processing vote results and resolving conflicts

This is the first coordinator agent in the pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
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

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import DeliberationCoordinator, VoteType


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
        """Validate extractions using LLM, batching to respect token limits."""
        if not entities and not triples:
            return {"overall_quality": 0.0}

        # Process in batches of 40 items to stay within token limits
        BATCH = 40
        all_entity_validations: List[Dict[str, Any]] = []
        all_triple_validations: List[Dict[str, Any]] = []
        quality_scores: List[float] = []
        all_recommendations: List[str] = []

        entity_batches = [entities[i:i + BATCH] for i in range(0, max(len(entities), 1), BATCH)] if entities else [[]]
        triple_batches = [triples[i:i + BATCH] for i in range(0, max(len(triples), 1), BATCH)] if triples else [[]]

        # Pair up entity and triple batches
        n = max(len(entity_batches), len(triple_batches))
        for idx in range(n):
            e_batch = entity_batches[idx] if idx < len(entity_batches) else []
            t_batch = triple_batches[idx] if idx < len(triple_batches) else []
            if not e_batch and not t_batch:
                continue

            entities_json = json.dumps(e_batch, indent=2)
            triples_json = json.dumps(t_batch, indent=2)

            prompt = VALIDATION_PROMPT.format(
                text=text[:6000],
                entities_json=entities_json,
                triples_json=triples_json,
                domain=domain or "general",
            )

            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert extraction validator. Be thorough but fair in assessment.",
                tier=ModelTier.LARGE,
                max_tokens=4096,
            )

            all_entity_validations.extend(result.get("entity_validations", []))
            all_triple_validations.extend(result.get("triple_validations", []))
            quality_scores.append(result.get("overall_quality", 0.0))
            all_recommendations.extend(result.get("recommendations", []))

        return {
            "entity_validations": all_entity_validations,
            "triple_validations": all_triple_validations,
            "overall_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "recommendations": all_recommendations,
        }

    def _refine_extractions(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        validation: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Refine extractions based on validation feedback, in batches."""
        BATCH = 30
        all_refined_entities: List[Dict[str, Any]] = []
        all_refined_triples: List[Dict[str, Any]] = []
        quality_scores: List[float] = []

        entity_batches = [entities[i:i + BATCH] for i in range(0, max(len(entities), 1), BATCH)] if entities else [[]]
        triple_batches = [triples[i:i + BATCH] for i in range(0, max(len(triples), 1), BATCH)] if triples else [[]]

        n = max(len(entity_batches), len(triple_batches))
        for idx in range(n):
            e_batch = entity_batches[idx] if idx < len(entity_batches) else []
            t_batch = triple_batches[idx] if idx < len(triple_batches) else []
            if not e_batch and not t_batch:
                continue

            extractions_json = json.dumps({
                "entities": e_batch,
                "triples": t_batch,
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
                tier=ModelTier.MEDIUM,  # Was LARGE (gpt-4o) — downgraded to reduce quota burn
                max_tokens=4096,
            )

            all_refined_entities.extend(result.get("refined_entities", []))
            all_refined_triples.extend(result.get("refined_triples", []))
            quality_scores.append(result.get("quality_after_refinement", 0.0))

        return {
            "refined_entities": all_refined_entities if all_refined_entities else entities,
            "refined_triples": all_refined_triples if all_refined_triples else triples,
            "quality_after_refinement": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        }

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

    # ==================== Deliberation Methods ====================

    def process_pending_deliberations(self) -> Dict[str, Any]:
        """
        Process all pending deliberations and resolve them.
        
        This method:
        1. Checks for pending hypotheses that need votes
        2. Triggers agent voting if needed
        3. Initiates debates for conflicting votes
        4. Resolves deliberations after sufficient votes/debate
        
        Returns:
            Summary of deliberation processing
        """
        if not self._deliberation_coordinator:
            return {"status": "no_coordinator"}
        
        # Process any pending vote requests first
        self.process_vote_requests()
        
        # Check for debate requests
        debate_requests = self.check_for_debate_requests()
        for req in debate_requests:
            self._participate_in_debate(req)
        
        # Process pending hypotheses that are ready for decision
        resolved = self._deliberation_coordinator.process_pending(max_wait_seconds=2.0)
        
        # Get accepted and rejected hypotheses
        accepted = self._deliberation_coordinator.get_accepted_hypotheses()
        rejected = self._deliberation_coordinator.get_rejected_hypotheses()
        
        return {
            "resolved_this_round": len(resolved),
            "total_accepted": len(accepted),
            "total_rejected": len(rejected),
            "stats": self._deliberation_coordinator.get_stats(),
        }

    def _participate_in_debate(self, debate_request: Dict[str, Any]) -> None:
        """
        Participate in a debate by providing arguments.
        
        As a coordinator, we analyze the current votes and provide
        a reasoned argument for or against the hypothesis.
        """
        hypothesis_id = debate_request.get("hypothesis_id")
        content = debate_request.get("content", {})
        current_votes = debate_request.get("current_votes", {})
        
        # Analyze the hypothesis
        accepts = current_votes.get("accepts", [])
        rejects = current_votes.get("rejects", [])
        
        # Generate our argument using LLM
        prompt = f"""A hypothesis is being debated by multiple agents.

HYPOTHESIS:
{json.dumps(content, indent=2)}

ARGUMENTS FOR ACCEPTANCE:
{json.dumps([v.get("rationale") for v in accepts], indent=2)}

ARGUMENTS FOR REJECTION:
{json.dumps([v.get("rationale") for v in rejects], indent=2)}

As the Extraction Validator, analyze these arguments and provide:
1. Your position (support or oppose)
2. Your reasoning
3. Any evidence you can add

Return JSON:
{{
    "position": "support" or "oppose",
    "argument": "<your detailed argument>",
    "key_points": ["<point1>", "<point2>"]
}}"""

        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert validator. Provide a well-reasoned argument.",
            tier=ModelTier.LARGE,
            max_tokens=4096,
        )
        
        position = result.get("position", "support")
        argument = result.get("argument", "No detailed argument provided")
        
        self.provide_debate_argument(
            hypothesis_id=hypothesis_id,
            position=position,
            argument=argument,
        )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        ExtractionValidator's voting logic.
        
        As a coordinator, we use LLM to provide high-quality votes.
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        # Use LLM to evaluate
        prompt = f"""Evaluate this {hypothesis_type} hypothesis for validity.

HYPOTHESIS:
{json.dumps(hypothesis_content, indent=2)}

Assess:
1. Is this a valid extraction?
2. What is your confidence?
3. What issues or concerns do you have?

Return JSON:
{{
    "valid": <true/false>,
    "confidence": <0.0-1.0>,
    "rationale": "<your reasoning>",
    "issues": ["<issue1>", ...]
}}"""

        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert extraction validator.",
            tier=ModelTier.LARGE,
            max_tokens=4096,
        )
        
        valid = result.get("valid", False)
        confidence = result.get("confidence", 0.5)
        rationale = result.get("rationale", "No rationale provided")
        
        if valid and confidence >= 0.8:
            return VoteType.STRONG_ACCEPT, confidence, rationale
        elif valid and confidence >= 0.6:
            return VoteType.ACCEPT, confidence, rationale
        elif valid:
            return VoteType.WEAK_ACCEPT, confidence, rationale
        elif confidence <= 0.2:
            return VoteType.STRONG_REJECT, confidence, rationale
        elif confidence <= 0.4:
            return VoteType.REJECT, confidence, rationale
        else:
            return VoteType.WEAK_REJECT, confidence, rationale

    def get_deliberation_results(self) -> Dict[str, Any]:
        """
        Get all deliberation results for integration.
        
        Returns:
            Dict with accepted entities and triples from deliberation
        """
        if not self._deliberation_coordinator:
            return {"entities": [], "triples": []}
        
        accepted = self._deliberation_coordinator.get_accepted_hypotheses()
        
        entities = []
        triples = []
        
        for hyp in accepted:
            if hyp.hypothesis_type == "entity":
                entity = hyp.content.copy()
                entity["deliberation_confidence"] = hyp.final_confidence
                entity["deliberation_id"] = hyp.id
                entities.append(entity)
            elif hyp.hypothesis_type in ["triple", "relation"]:
                triple = hyp.content.copy()
                triple["deliberation_confidence"] = hyp.final_confidence
                triple["deliberation_id"] = hyp.id
                triples.append(triple)
        
        return {
            "entities": entities,
            "triples": triples,
            "stats": self._deliberation_coordinator.get_stats() if self._deliberation_coordinator else {},
        }
