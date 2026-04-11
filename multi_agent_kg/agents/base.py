"""
Enhanced Base Agent with Full Memory, Communication, and Deliberation Integration.

All agents inherit from this base class which provides:
- SharedMemory access (episodic, semantic, working memory + blackboard)
- MessageBus communication (inter-agent messaging, voting, delegation)
- Multi-agent deliberation (hypothesis voting, debate, consensus)
- Confidence calculation with self-consistency
- Iterative refinement support
- LLM interaction with tiered model selection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from enum import Enum
import json
import os

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory, MemoryType, BlackboardEntry
from multi_agent_kg.core.communication import (
    MessageBus, 
    CollaborationProtocol, 
    CommunicationType,
    MessagePriority,
    AgentMessage,
)
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion_json

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import DeliberationCoordinator, VoteType


class AgentRole(str, Enum):
    """Role of the agent in the pipeline."""
    WORKER = "worker"           # Performs extraction/processing
    COORDINATOR = "coordinator" # Orchestrates and validates


class ModelTier(str, Enum):
    """Model tier for tiered model selection."""
    SMALL = "small"     # ~4B params - fast (qwen3:4b)
    MEDIUM = "medium"   # balanced reasoning
    LARGE = "large"     # highest quality reasoning


# Default model mapping (Ollama models on GPU via SSH tunnel)
DEFAULT_MODEL_TIERS = {
    ModelTier.SMALL: os.getenv("LLM_SMALL_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
    ModelTier.MEDIUM: os.getenv("LLM_MEDIUM_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
    ModelTier.LARGE: os.getenv("LLM_LARGE_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
}


@dataclass
class ExtractionResult:
    """Result from an extraction operation."""
    items: List[Dict[str, Any]]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None


@dataclass
class AgentContext:
    """Context passed to agents during processing."""
    document_id: str
    text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    domain: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 4
    quality_threshold: float = 0.85
    previous_feedback: List[Dict[str, Any]] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Enhanced base agent with full memory and communication integration.
    
    All agents in the system inherit from this class and get:
    - Automatic memory integration
    - Inter-agent communication via MessageBus
    - Multi-agent deliberation with voting and debate
    - Blackboard-based hypothesis posting and voting
    - Self-consistency based confidence calculation
    - Iterative refinement with feedback loops
    - Tiered model selection based on task complexity
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        model_tiers: Optional[Dict[ModelTier, str]] = None,
        default_tier: ModelTier = ModelTier.MEDIUM,
        quality_threshold: float = 0.85,
        max_iterations: int = 4,
    ):
        """
        Initialize the enhanced base agent.
        
        Args:
            name: Unique agent name
            role: Worker or coordinator
            knowledge_graph: Shared knowledge graph
            shared_memory: Shared memory system
            message_bus: Inter-agent communication bus
            llm_config: Base LLM configuration
            model_tiers: Mapping of tier to model name
            default_tier: Default model tier to use
            quality_threshold: Threshold for acceptable quality
            max_iterations: Maximum refinement iterations
        """
        self.name = name
        self.role = role
        self.knowledge_graph = knowledge_graph
        self.shared_memory = shared_memory
        self.message_bus = message_bus
        self.llm_config = llm_config or LLMConfig()
        self.model_tiers = model_tiers or DEFAULT_MODEL_TIERS
        self.default_tier = default_tier
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        
        # Collaboration protocol for structured communication
        self.collab = CollaborationProtocol(message_bus) if message_bus else None
        
        # Deliberation coordinator (set by orchestrator)
        self._deliberation_coordinator: Optional["DeliberationCoordinator"] = None
        
        # Message subscriptions
        self._subscriptions: List[str] = []
        
        # Statistics
        self.stats = {
            "calls": 0,
            "llm_calls": 0,
            "escalations": 0,
            "refinements": 0,
            "deliberations_submitted": 0,
            "votes_cast": 0,
            "debates_participated": 0,
        }

    def set_deliberation_coordinator(self, coordinator: "DeliberationCoordinator") -> None:
        """Set the deliberation coordinator for multi-agent voting."""
        self._deliberation_coordinator = coordinator

    @abstractmethod
    def run(self, context: AgentContext, **kwargs) -> ExtractionResult:
        """
        Execute the agent's main task.
        
        Args:
            context: Processing context with document, entities, etc.
            **kwargs: Additional agent-specific arguments
            
        Returns:
            ExtractionResult with items, confidence, and metadata
        """
        pass

    # ==================== LLM Methods ====================

    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Call the LLM with the specified tier.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            tier: Model tier (defaults to self.default_tier)
            temperature: Override temperature
            max_tokens: Override max tokens
            response_format: JSON schema for structured output
            
        Returns:
            Parsed JSON response from LLM
        """
        tier = tier or self.default_tier
        model = self.model_tiers.get(tier, self.llm_config.model)
        
        config = LLMConfig(
            model=model,
            temperature=temperature if temperature is not None else self.llm_config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.llm_config.max_tokens,
        )
        
        self.stats["llm_calls"] += 1
        
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return chat_completion_json(
            messages=messages,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def call_llm_with_self_consistency(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        n_samples: int = 3,
        temperature: float = 0.7,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Call LLM multiple times and compute confidence via majority voting.
        
        This implements self-consistency for confidence estimation:
        - Run the same prompt n_samples times with temperature > 0
        - Compute agreement between responses
        - Confidence = agreement ratio
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            tier: Model tier
            n_samples: Number of samples to generate
            temperature: Temperature for sampling (should be > 0)
            
        Returns:
            Tuple of (consensus result, confidence score)
        """
        responses = []
        for _ in range(n_samples):
            try:
                response = self.call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    tier=tier,
                    temperature=temperature,
                )
                responses.append(response)
            except Exception as e:
                self.log(f"Self-consistency sample failed: {e}", level="WARNING")
        
        if not responses:
            return {}, 0.0
        
        # Compute consensus
        consensus, confidence = self._compute_consensus(responses)
        return consensus, confidence

    def _compute_consensus(
        self,
        responses: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Compute consensus from multiple responses.
        
        Uses a simple voting mechanism:
        - Serialize each response
        - Count occurrences
        - Return most common with frequency as confidence
        """
        if not responses:
            return {}, 0.0
        
        # Serialize for comparison
        serialized = []
        for r in responses:
            try:
                serialized.append(json.dumps(r, sort_keys=True))
            except:
                serialized.append(str(r))
        
        # Count votes
        from collections import Counter
        counts = Counter(serialized)
        most_common, count = counts.most_common(1)[0]
        
        confidence = count / len(responses)
        
        # Deserialize winner
        try:
            result = json.loads(most_common)
        except:
            result = responses[0]
        
        return result, confidence

    # ==================== Memory Methods ====================

    def store_in_memory(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store data in shared memory.
        
        Args:
            memory_type: Type of memory (episodic, semantic, working)
            content: Content to store
            references: IDs of related memory entries
            metadata: Additional metadata
            
        Returns:
            Memory entry ID if stored, None if no memory available
        """
        if not self.shared_memory:
            return None
        
        return self.shared_memory.store(
            memory_type=memory_type,
            content=content,
            source=self.name,
            references=references,
            metadata=metadata,
        )

    def retrieve_from_memory(
        self,
        memory_type: Optional[MemoryType] = None,
        entity: Optional[str] = None,
        limit: int = 20,
    ) -> List[Any]:
        """
        Retrieve data from shared memory.
        
        Args:
            memory_type: Filter by memory type
            entity: Filter by entity mention
            limit: Maximum entries to return
            
        Returns:
            List of matching memory entries
        """
        if not self.shared_memory:
            return []
        
        return self.shared_memory.retrieve(
            memory_type=memory_type,
            entity=entity,
            limit=limit,
        )

    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """Get cross-document context for an entity."""
        if not self.shared_memory:
            return {}
        return self.shared_memory.get_entity_context(entity_id)

    # ==================== Blackboard Methods ====================

    def post_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        confidence: float,
        evidence: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Post a hypothesis to the blackboard for other agents to vote on.
        
        Args:
            hypothesis: The hypothesis (entity, triple, etc.)
            confidence: Initial confidence score
            evidence: Supporting evidence
            
        Returns:
            Blackboard entry ID if posted
        """
        if not self.shared_memory:
            return None
        
        return self.shared_memory.post_to_blackboard(
            author=self.name,
            entry_type="hypothesis",
            content={
                "hypothesis": hypothesis,
                "confidence": confidence,
                "evidence": evidence or [],
            },
        )

    def vote_on_hypothesis(
        self,
        entry_id: str,
        vote: float,
        rationale: Optional[str] = None,
    ) -> None:
        """
        Vote on a blackboard hypothesis.
        
        Args:
            entry_id: Blackboard entry ID
            vote: Confidence vote (0-1)
            rationale: Reason for vote
        """
        if not self.shared_memory:
            return
        
        self.shared_memory.vote_on_blackboard(
            entry_id=entry_id,
            voter=self.name,
            confidence=vote,
        )
        
        if rationale:
            self.shared_memory.respond_to_blackboard(
                entry_id=entry_id,
                responder=self.name,
                response_type="vote_rationale",
                response_content={"rationale": rationale},
            )

    def get_pending_hypotheses(
        self,
        min_votes: int = 0,
    ) -> List[BlackboardEntry]:
        """Get pending hypotheses that need votes."""
        if not self.shared_memory:
            return []
        
        return self.shared_memory.get_blackboard_entries(
            entry_type="hypothesis",
            status="pending",
            min_votes=min_votes,
        )

    def resolve_hypothesis(
        self,
        entry_id: str,
        accepted: bool,
    ) -> None:
        """Mark a hypothesis as accepted or rejected."""
        if not self.shared_memory:
            return
        
        status = "accepted" if accepted else "rejected"
        self.shared_memory.resolve_blackboard_entry(entry_id, status)

    # ==================== Communication Methods ====================

    def send_message(
        self,
        receiver: str,
        comm_type: CommunicationType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_response: bool = False,
    ) -> Optional[str]:
        """
        Send a message to another agent.
        
        Args:
            receiver: Target agent name
            comm_type: Type of communication
            content: Message content
            priority: Message priority
            requires_response: Whether a response is expected
            
        Returns:
            Message ID if sent
        """
        if not self.message_bus:
            return None
        
        return self.message_bus.send(
            sender=self.name,
            receiver=receiver,
            comm_type=comm_type,
            content=content,
            priority=priority,
            requires_response=requires_response,
        )

    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages for this agent."""
        if not self.message_bus:
            return []
        
        return self.message_bus.receive(self.name)

    def request_refinement(
        self,
        target_agent: str,
        item: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Request another agent to refine an extraction.
        
        Args:
            target_agent: Agent to do the refinement
            item: Item to refine
            context: Additional context
            
        Returns:
            Request message ID
        """
        if not self.collab:
            return None
        
        self.stats["refinements"] += 1
        return self.collab.request_refinement(
            requester=self.name,
            target_agent=target_agent,
            item=item,
            context=context,
        )

    def delegate_task(
        self,
        delegate: str,
        task: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Optional[str]:
        """
        Delegate a task to another agent.
        
        Args:
            delegate: Agent to perform the task
            task: Task specification
            priority: Task priority
            
        Returns:
            Delegation message ID
        """
        if not self.collab:
            return None
        
        return self.collab.delegate_task(
            delegator=self.name,
            delegate=delegate,
            task=task,
            priority=priority,
        )

    def provide_feedback(
        self,
        recipient: str,
        original_output: Dict[str, Any],
        feedback: Dict[str, Any],
        suggested_corrections: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """
        Provide feedback on another agent's output.
        
        Args:
            recipient: Agent receiving feedback
            original_output: The output being critiqued
            feedback: Feedback content
            suggested_corrections: Suggested fixes
            
        Returns:
            Feedback message ID
        """
        if not self.collab:
            return None
        
        return self.collab.provide_feedback(
            provider=self.name,
            recipient=recipient,
            original_output=original_output,
            feedback=feedback,
            suggested_corrections=suggested_corrections,
        )

    # ==================== Escalation Methods ====================

    def escalate_to_coordinator(
        self,
        reason: str,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Escalate low-confidence items to coordinator for deliberation.
        
        This is used when:
        - Confidence is below threshold after self-consistency
        - Conflicting extractions are detected
        - Domain-specific rules are unclear
        
        Args:
            reason: Why escalation is needed
            items: Items needing review
            context: Relevant context
            
        Returns:
            Escalation message ID
        """
        self.stats["escalations"] += 1
        
        # Post to blackboard
        if self.shared_memory:
            self.shared_memory.post_to_blackboard(
                author=self.name,
                entry_type="escalation",
                content={
                    "reason": reason,
                    "items": items,
                    "context": context,
                },
            )
        
        # Also send message to coordinator
        if self.message_bus:
            return self.message_bus.send(
                sender=self.name,
                receiver="ExtractionValidator",
                comm_type=CommunicationType.DELEGATE,
                content={
                    "action": "escalation",
                    "reason": reason,
                    "items": items,
                    "context": context,
                },
                priority=MessagePriority.HIGH,
                requires_response=True,
            )
        
        return None

    # ==================== Deliberation Methods ====================

    def submit_for_deliberation(
        self,
        hypothesis_type: str,
        content: Dict[str, Any],
        confidence: float,
        evidence: Optional[List[str]] = None,
        document_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Submit a hypothesis for multi-agent deliberation.
        
        Use this when confidence is below threshold and you want
        other agents to vote on the extraction.
        
        Args:
            hypothesis_type: Type (entity, relation, triple)
            content: The hypothesis content
            confidence: Your confidence in this hypothesis
            evidence: Supporting evidence
            document_id: Source document
            
        Returns:
            Hypothesis ID if submitted
        """
        if not self._deliberation_coordinator:
            # Fall back to blackboard posting
            if self.shared_memory:
                return self.shared_memory.post_to_blackboard(
                    author=self.name,
                    entry_type="hypothesis",
                    content={
                        "type": hypothesis_type,
                        "content": content,
                        "confidence": confidence,
                        "evidence": evidence or [],
                    },
                )
            return None
        
        self.stats["deliberations_submitted"] += 1
        return self._deliberation_coordinator.submit_hypothesis(
            author=self.name,
            hypothesis_type=hypothesis_type,
            content=content,
            confidence=confidence,
            evidence=evidence,
            document_id=document_id,
        )

    def cast_vote(
        self,
        hypothesis_id: str,
        vote_type: "VoteType",
        confidence: float,
        rationale: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """
        Vote on another agent's hypothesis.
        
        Args:
            hypothesis_id: ID of hypothesis to vote on
            vote_type: Your vote (from VoteType enum)
            confidence: Your confidence in this vote (0-1)
            rationale: Reason for your vote
            evidence: Supporting evidence
        """
        if not self._deliberation_coordinator:
            # Fall back to blackboard voting
            if self.shared_memory:
                self.shared_memory.vote_on_blackboard(
                    entry_id=hypothesis_id,
                    voter=self.name,
                    confidence=confidence,
                )
            return
        
        self.stats["votes_cast"] += 1
        self._deliberation_coordinator.receive_vote(
            hypothesis_id=hypothesis_id,
            voter=self.name,
            vote_type=vote_type,
            confidence=confidence,
            rationale=rationale,
            evidence=evidence,
        )

    def provide_debate_argument(
        self,
        hypothesis_id: str,
        position: str,
        argument: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """
        Provide an argument in a debate.
        
        Args:
            hypothesis_id: ID of hypothesis being debated
            position: "support" or "oppose"
            argument: Your argument
            evidence: Supporting evidence
        """
        if not self._deliberation_coordinator:
            return
        
        self.stats["debates_participated"] += 1
        self._deliberation_coordinator.receive_debate_argument(
            hypothesis_id=hypothesis_id,
            agent=self.name,
            position=position,
            argument=argument,
            evidence=evidence,
        )

    def check_for_vote_requests(self) -> List[Dict[str, Any]]:
        """
        Check for pending vote requests from the deliberation coordinator.
        
        Returns:
            List of vote request contents
        """
        if not self.message_bus:
            return []
        
        messages = self.message_bus.receive(self.name)
        vote_requests = [
            m.content for m in messages
            if m.comm_type == CommunicationType.REQUEST
            and m.content.get("action") == "vote"
        ]
        
        return vote_requests

    def check_for_debate_requests(self) -> List[Dict[str, Any]]:
        """
        Check for pending debate requests from the deliberation coordinator.
        
        Returns:
            List of debate request contents
        """
        if not self.message_bus:
            return []
        
        messages = self.message_bus.receive(self.name)
        debate_requests = [
            m.content for m in messages
            if m.comm_type == CommunicationType.REQUEST
            and m.content.get("action") == "debate"
        ]
        
        return debate_requests

    def process_vote_requests(self, context: Optional[AgentContext] = None) -> int:
        """
        Process all pending vote requests.
        
        This method checks for vote requests, evaluates each hypothesis,
        and submits votes. Subclasses can override evaluate_hypothesis_for_vote
        to provide agent-specific voting logic.
        
        Args:
            context: Optional processing context
            
        Returns:
            Number of votes cast
        """
        vote_requests = self.check_for_vote_requests()
        votes_cast = 0
        
        for request in vote_requests:
            hypothesis_id = request.get("hypothesis_id")
            hypothesis_type = request.get("hypothesis_type")
            content = request.get("content")
            
            if not hypothesis_id or not content:
                continue
            
            # Evaluate and vote
            vote_type, confidence, rationale = self.evaluate_hypothesis_for_vote(
                hypothesis_content=content,
                hypothesis_type=hypothesis_type,
                context=context,
            )
            
            # Import here to avoid circular import
            from multi_agent_kg.core.deliberation import VoteType
            
            if vote_type != VoteType.ABSTAIN:
                self.cast_vote(
                    hypothesis_id=hypothesis_id,
                    vote_type=vote_type,
                    confidence=confidence,
                    rationale=rationale,
                )
                votes_cast += 1
        
        return votes_cast

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple["VoteType", float, str]:
        """
        Evaluate a hypothesis and determine your vote.
        
        Override this method in subclasses to implement
        agent-specific voting logic.
        
        Args:
            hypothesis_content: The hypothesis to evaluate
            hypothesis_type: Type of hypothesis (entity, relation, triple)
            context: Optional processing context
            
        Returns:
            Tuple of (vote_type, confidence, rationale)
        """
        # Default implementation - abstain
        from multi_agent_kg.core.deliberation import VoteType
        return VoteType.ABSTAIN, 0.5, "No specific evaluation logic for this agent"

    # ==================== Utility Methods ====================

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message from this agent."""
        print(f"[{level}] [{self.name}] {msg}")

    def should_escalate(self, confidence: float) -> bool:
        """Check if confidence is below threshold for escalation."""
        return confidence < self.quality_threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "role": self.role.value,
            **self.stats,
        }
