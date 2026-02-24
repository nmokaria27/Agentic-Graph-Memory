"""
Multi-Agent Deliberation System.

Implements the novel KARMA-inspired deliberation mechanisms:
- Hypothesis posting to blackboard
- Multi-agent voting with weighted confidence
- Debate loops for conflicting hypotheses
- Consensus resolution with rationale tracking
- Cross-agent validation and feedback

This module is the core of the multi-agent collaboration system,
enabling agents to deliberate on uncertain extractions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum
import statistics

from multi_agent_kg.core.memory import SharedMemory, BlackboardEntry
from multi_agent_kg.core.communication import (
    MessageBus,
    CommunicationType,
    MessagePriority,
    AgentMessage,
)


class VoteType(str, Enum):
    """Types of votes agents can cast."""
    STRONG_ACCEPT = "strong_accept"    # High confidence accept (weight 1.0)
    ACCEPT = "accept"                   # Accept (weight 0.75)
    WEAK_ACCEPT = "weak_accept"        # Tentative accept (weight 0.5)
    ABSTAIN = "abstain"                 # No opinion (weight 0)
    WEAK_REJECT = "weak_reject"        # Tentative reject (weight -0.5)
    REJECT = "reject"                   # Reject (weight -0.75)
    STRONG_REJECT = "strong_reject"    # High confidence reject (weight -1.0)


VOTE_WEIGHTS = {
    VoteType.STRONG_ACCEPT: 1.0,
    VoteType.ACCEPT: 0.75,
    VoteType.WEAK_ACCEPT: 0.5,
    VoteType.ABSTAIN: 0.0,
    VoteType.WEAK_REJECT: -0.5,
    VoteType.REJECT: -0.75,
    VoteType.STRONG_REJECT: -1.0,
}


class DeliberationStatus(str, Enum):
    """Status of a deliberation."""
    PENDING = "pending"           # Awaiting votes
    VOTING = "voting"             # Voting in progress
    DEBATING = "debating"         # In debate phase
    CONSENSUS = "consensus"       # Consensus reached
    ACCEPTED = "accepted"         # Hypothesis accepted
    REJECTED = "rejected"         # Hypothesis rejected
    DEADLOCK = "deadlock"         # No consensus possible


@dataclass
class Vote:
    """A vote cast by an agent on a hypothesis."""
    voter: str
    vote_type: VoteType
    confidence: float  # Agent's confidence in their vote (0-1)
    rationale: str
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted score: vote_weight * confidence."""
        return VOTE_WEIGHTS[self.vote_type] * self.confidence


@dataclass
class DebateArgument:
    """An argument made during a debate."""
    agent: str
    position: str  # "support" or "oppose"
    argument: str
    evidence: List[str] = field(default_factory=list)
    counter_to: Optional[str] = None  # ID of argument being countered
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Hypothesis:
    """A hypothesis submitted for deliberation."""
    id: str
    author: str
    hypothesis_type: str  # "entity", "relation", "triple"
    content: Dict[str, Any]
    initial_confidence: float
    evidence: List[str] = field(default_factory=list)
    document_id: Optional[str] = None
    
    # Deliberation state
    status: DeliberationStatus = DeliberationStatus.PENDING
    votes: Dict[str, Vote] = field(default_factory=dict)
    debate_arguments: List[DebateArgument] = field(default_factory=list)
    final_confidence: Optional[float] = None
    resolution_rationale: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    def add_vote(self, vote: Vote) -> None:
        """Add a vote to this hypothesis."""
        self.votes[vote.voter] = vote
        if self.status == DeliberationStatus.PENDING:
            self.status = DeliberationStatus.VOTING
    
    def add_argument(self, argument: DebateArgument) -> None:
        """Add a debate argument."""
        self.debate_arguments.append(argument)
        if self.status == DeliberationStatus.VOTING:
            self.status = DeliberationStatus.DEBATING
    
    def get_vote_summary(self) -> Dict[str, Any]:
        """Get summary of votes."""
        if not self.votes:
            return {"total": 0, "weighted_score": 0, "mean_confidence": 0}
        
        votes_list = list(self.votes.values())
        weighted_scores = [v.weighted_score for v in votes_list]
        
        return {
            "total": len(votes_list),
            "weighted_score": sum(weighted_scores),
            "normalized_score": sum(weighted_scores) / len(votes_list),
            "mean_confidence": statistics.mean([v.confidence for v in votes_list]),
            "accepts": sum(1 for v in votes_list if v.vote_type in [VoteType.STRONG_ACCEPT, VoteType.ACCEPT, VoteType.WEAK_ACCEPT]),
            "rejects": sum(1 for v in votes_list if v.vote_type in [VoteType.STRONG_REJECT, VoteType.REJECT, VoteType.WEAK_REJECT]),
            "abstains": sum(1 for v in votes_list if v.vote_type == VoteType.ABSTAIN),
        }


class DeliberationCoordinator:
    """
    Coordinates multi-agent deliberation on hypotheses.
    
    This is the central hub for:
    1. Receiving hypotheses from worker agents
    2. Soliciting votes from relevant agents
    3. Detecting conflicts and initiating debates
    4. Resolving consensus or deadlock
    5. Communicating decisions back to agents
    
    Flow:
    1. Agent posts hypothesis → stored in deliberation queue
    2. Coordinator broadcasts "vote_request" to relevant agents
    3. Agents vote with rationales
    4. If votes conflict → debate phase
    5. Coordinator resolves based on weighted consensus
    6. Result posted to blackboard and communicated to agents
    """
    
    # Agent weights for voting (coordinators have higher weight)
    AGENT_WEIGHTS = {
        "EntityExtractor": 1.0,
        "RelationExtractor": 1.0,
        "EvidenceLinker": 1.2,  # Evidence expert gets slight boost
        "ExtractionValidator": 1.5,  # Coordinator has higher weight
        "ExtractionVerificationAgent": 1.5,
        "KnowledgeOrganizer": 1.3,
        "DomainClassifier": 0.8,
        "DocumentProcessor": 0.5,
    }
    
    # Thresholds for decision making
    CONSENSUS_THRESHOLD = 0.6  # Normalized score needed for consensus
    CONFLICT_THRESHOLD = 0.3  # Below this absolute score triggers debate
    MIN_VOTES_FOR_DECISION = 2  # Minimum votes needed
    MAX_DEBATE_ROUNDS = 3  # Maximum debate iterations
    
    def __init__(
        self,
        shared_memory: SharedMemory,
        message_bus: MessageBus,
        voting_agents: Optional[List[str]] = None,
        consensus_threshold: float = 0.6,
        min_votes: int = 2,
        debug_logger=None,
    ):
        """
        Initialize the deliberation coordinator.
        
        Args:
            shared_memory: Shared memory for blackboard access
            message_bus: Message bus for agent communication
            voting_agents: List of agents that can vote (default: all extractors)
            consensus_threshold: Threshold for consensus decision
            min_votes: Minimum votes required for a decision
            debug_logger: Debug logger for tracking votes and debates
        """
        self.memory = shared_memory
        self.bus = message_bus
        self.voting_agents = voting_agents or [
            "EntityExtractor",
            "RelationExtractor", 
            "EvidenceLinker",
        ]
        self.consensus_threshold = consensus_threshold
        self.min_votes = min_votes
        self.debug_logger = debug_logger
        
        # Active deliberations
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.pending_queue: List[str] = []  # IDs of hypotheses awaiting votes
        
        # Statistics
        self.stats = {
            "hypotheses_received": 0,
            "votes_collected": 0,
            "debates_initiated": 0,
            "consensus_reached": 0,
            "deadlocks": 0,
            "accepted": 0,
            "rejected": 0,
        }
    
    def submit_hypothesis(
        self,
        author: str,
        hypothesis_type: str,
        content: Dict[str, Any],
        confidence: float,
        evidence: Optional[List[str]] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Submit a hypothesis for deliberation.
        
        Args:
            author: Agent submitting the hypothesis
            hypothesis_type: Type of hypothesis (entity, relation, triple)
            content: The actual hypothesis content
            confidence: Author's confidence in the hypothesis
            evidence: Supporting evidence
            document_id: Source document ID
            
        Returns:
            Hypothesis ID
        """
        hyp_id = f"hyp_{author}_{datetime.now().timestamp()}"
        
        hypothesis = Hypothesis(
            id=hyp_id,
            author=author,
            hypothesis_type=hypothesis_type,
            content=content,
            initial_confidence=confidence,
            evidence=evidence or [],
            document_id=document_id,
        )
        
        self.hypotheses[hyp_id] = hypothesis
        self.pending_queue.append(hyp_id)
        self.stats["hypotheses_received"] += 1
        
        # Post to blackboard
        self.memory.post_to_blackboard(
            author=author,
            entry_type="hypothesis",
            content={
                "hypothesis_id": hyp_id,
                "hypothesis_type": hypothesis_type,
                "content": content,
                "confidence": confidence,
                "evidence": evidence or [],
            },
        )
        
        # Broadcast vote request
        self._request_votes(hyp_id)
        
        return hyp_id
    
    def _request_votes(self, hypothesis_id: str) -> None:
        """Request votes from voting agents."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        # Don't request vote from the author
        voters = [a for a in self.voting_agents if a != hypothesis.author]
        
        for agent in voters:
            self.bus.send(
                sender="DeliberationCoordinator",
                receiver=agent,
                comm_type=CommunicationType.REQUEST,
                content={
                    "action": "vote",
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_type": hypothesis.hypothesis_type,
                    "content": hypothesis.content,
                    "author": hypothesis.author,
                    "initial_confidence": hypothesis.initial_confidence,
                    "evidence": hypothesis.evidence,
                },
                priority=MessagePriority.HIGH,
                requires_response=True,
            )
    
    def receive_vote(
        self,
        hypothesis_id: str,
        voter: str,
        vote_type: VoteType,
        confidence: float,
        rationale: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """
        Receive a vote on a hypothesis.
        
        Args:
            hypothesis_id: ID of hypothesis being voted on
            voter: Agent casting the vote
            vote_type: Type of vote
            confidence: Voter's confidence in their vote
            rationale: Reason for the vote
            evidence: Supporting evidence for the vote
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        # Apply agent weight
        agent_weight = self.AGENT_WEIGHTS.get(voter, 1.0)
        adjusted_confidence = confidence * agent_weight
        
        vote = Vote(
            voter=voter,
            vote_type=vote_type,
            confidence=min(adjusted_confidence, 1.0),
            rationale=rationale,
            evidence=evidence or [],
        )
        
        hypothesis.add_vote(vote)
        self.stats["votes_collected"] += 1

        # Only debug-log votes when verbose debug is explicitly enabled;
        # the default synchronous rule-based voting fires hundreds of times
        # per pipeline run and floods stdout with identical lines.
        if self.debug_logger and getattr(self.debug_logger, "verbose_votes", False):
            self.debug_logger.log_vote(
                voter=voter,
                hypothesis_id=hypothesis_id,
                vote_type=vote_type.value,
                confidence=confidence,
                rationale=rationale,
                evidence=evidence,
                weighted_score=vote.weighted_score
            )
        
        # Also record on blackboard
        self.memory.vote_on_blackboard(
            entry_id=hypothesis_id,
            voter=voter,
            confidence=vote.weighted_score,
        )
        
        # Check if we can make a decision
        self._check_decision(hypothesis_id)
    
    def _check_decision(self, hypothesis_id: str) -> None:
        """Check if we can make a decision on a hypothesis."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        summary = hypothesis.get_vote_summary()
        
        # Need minimum votes
        if summary["total"] < self.min_votes:
            return
        
        normalized_score = summary["normalized_score"]
        
        # Clear consensus for accept
        if normalized_score >= self.consensus_threshold:
            self._resolve_hypothesis(
                hypothesis_id, 
                accepted=True,
                confidence=summary["mean_confidence"],
                rationale=f"Consensus accept (score: {normalized_score:.2f})",
            )
            return
        
        # Clear consensus for reject
        if normalized_score <= -self.consensus_threshold:
            self._resolve_hypothesis(
                hypothesis_id,
                accepted=False,
                confidence=summary["mean_confidence"],
                rationale=f"Consensus reject (score: {normalized_score:.2f})",
            )
            return
        
        # Conflict detected - check if we should debate
        if abs(normalized_score) < self.CONFLICT_THRESHOLD:
            if hypothesis.status != DeliberationStatus.DEBATING:
                self._initiate_debate(hypothesis_id)
    
    def _initiate_debate(self, hypothesis_id: str) -> None:
        """Initiate a debate on a conflicted hypothesis."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        hypothesis.status = DeliberationStatus.DEBATING
        self.stats["debates_initiated"] += 1
        
        # Get the conflicting votes
        accepts = [v for v in hypothesis.votes.values() 
                   if v.vote_type in [VoteType.STRONG_ACCEPT, VoteType.ACCEPT, VoteType.WEAK_ACCEPT]]
        rejects = [v for v in hypothesis.votes.values() 
                   if v.vote_type in [VoteType.STRONG_REJECT, VoteType.REJECT, VoteType.WEAK_REJECT]]
        
        # Request debate arguments
        self.bus.send(
            sender="DeliberationCoordinator",
            receiver="broadcast",
            comm_type=CommunicationType.REQUEST,
            content={
                "action": "debate",
                "hypothesis_id": hypothesis_id,
                "hypothesis_type": hypothesis.hypothesis_type,
                "content": hypothesis.content,
                "current_votes": {
                    "accepts": [{"voter": v.voter, "rationale": v.rationale} for v in accepts],
                    "rejects": [{"voter": v.voter, "rationale": v.rationale} for v in rejects],
                },
                "debate_prompt": "Please provide your argument for or against this hypothesis.",
            },
            priority=MessagePriority.HIGH,
            requires_response=True,
        )
    
    def receive_debate_argument(
        self,
        hypothesis_id: str,
        agent: str,
        position: str,
        argument: str,
        evidence: Optional[List[str]] = None,
        counter_to: Optional[str] = None,
    ) -> None:
        """
        Receive a debate argument from an agent.
        
        Args:
            hypothesis_id: ID of hypothesis being debated
            agent: Agent making the argument
            position: "support" or "oppose"
            argument: The argument text
            evidence: Supporting evidence
            counter_to: ID of argument being countered
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        debate_arg = DebateArgument(
            agent=agent,
            position=position,
            argument=argument,
            evidence=evidence or [],
            counter_to=counter_to,
        )
        
        hypothesis.add_argument(debate_arg)
        
        # Record in blackboard
        self.memory.respond_to_blackboard(
            entry_id=hypothesis_id,
            responder=agent,
            response_type="debate_argument",
            response_content={
                "position": position,
                "argument": argument,
                "evidence": evidence or [],
            },
        )
    
    def resolve_debate(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Resolve a debate and make a final decision.
        
        Uses weighted voting + argument quality assessment.
        
        Args:
            hypothesis_id: ID of hypothesis to resolve
            
        Returns:
            Resolution result
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return {"error": "Hypothesis not found"}
        
        summary = hypothesis.get_vote_summary()
        
        # Count debate arguments
        support_args = [a for a in hypothesis.debate_arguments if a.position == "support"]
        oppose_args = [a for a in hypothesis.debate_arguments if a.position == "oppose"]
        
        # Calculate final score with debate bonus
        base_score = summary["normalized_score"]
        debate_bonus = (len(support_args) - len(oppose_args)) * 0.1
        final_score = base_score + debate_bonus
        
        # Make decision
        if final_score > 0:
            accepted = True
            confidence = min(0.5 + abs(final_score) * 0.5, 0.95)
            rationale = f"Accepted after debate (final score: {final_score:.2f})"
        else:
            accepted = False
            confidence = min(0.5 + abs(final_score) * 0.5, 0.95)
            rationale = f"Rejected after debate (final score: {final_score:.2f})"
        
        self._resolve_hypothesis(hypothesis_id, accepted, confidence, rationale)
        
        return {
            "hypothesis_id": hypothesis_id,
            "accepted": accepted,
            "confidence": confidence,
            "rationale": rationale,
            "vote_summary": summary,
            "debate_arguments": len(hypothesis.debate_arguments),
        }
    
    def _resolve_hypothesis(
        self,
        hypothesis_id: str,
        accepted: bool,
        confidence: float,
        rationale: str,
    ) -> None:
        """Resolve a hypothesis with final decision."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        hypothesis.status = DeliberationStatus.ACCEPTED if accepted else DeliberationStatus.REJECTED
        hypothesis.final_confidence = confidence
        hypothesis.resolution_rationale = rationale
        hypothesis.resolved_at = datetime.now()
        
        # Update stats
        if accepted:
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        self.stats["consensus_reached"] += 1
        
        # Remove from pending queue
        if hypothesis_id in self.pending_queue:
            self.pending_queue.remove(hypothesis_id)
        
        # Update blackboard
        status = "accepted" if accepted else "rejected"
        self.memory.resolve_blackboard_entry(hypothesis_id, status)
        
        # Notify agents
        self.bus.send(
            sender="DeliberationCoordinator",
            receiver="broadcast",
            comm_type=CommunicationType.INFORM,
            content={
                "action": "hypothesis_resolved",
                "hypothesis_id": hypothesis_id,
                "accepted": accepted,
                "confidence": confidence,
                "rationale": rationale,
            },
            priority=MessagePriority.NORMAL,
        )
    
    def force_resolution(
        self,
        hypothesis_id: str,
        use_author_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Force resolution of a hypothesis when votes are insufficient.
        
        Args:
            hypothesis_id: Hypothesis to resolve
            use_author_confidence: Use author's original confidence
            
        Returns:
            Resolution result
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return {"error": "Hypothesis not found"}
        
        summary = hypothesis.get_vote_summary()
        
        if summary["total"] == 0:
            # No votes - use author's confidence
            if use_author_confidence and hypothesis.initial_confidence >= 0.5:
                self._resolve_hypothesis(
                    hypothesis_id,
                    accepted=True,
                    confidence=hypothesis.initial_confidence * 0.8,  # Discount
                    rationale="Accepted with author confidence (no votes)",
                )
                return {"accepted": True, "method": "author_confidence"}
            else:
                self._resolve_hypothesis(
                    hypothesis_id,
                    accepted=False,
                    confidence=0.5,
                    rationale="Rejected due to insufficient votes",
                )
                self.stats["deadlocks"] += 1
                return {"accepted": False, "method": "deadlock"}
        
        # Use available votes
        if summary["normalized_score"] >= 0:
            self._resolve_hypothesis(
                hypothesis_id,
                accepted=True,
                confidence=summary["mean_confidence"],
                rationale=f"Accepted with partial votes ({summary['total']} votes)",
            )
            return {"accepted": True, "method": "partial_consensus"}
        else:
            self._resolve_hypothesis(
                hypothesis_id,
                accepted=False,
                confidence=summary["mean_confidence"],
                rationale=f"Rejected with partial votes ({summary['total']} votes)",
            )
            return {"accepted": False, "method": "partial_consensus"}
    
    def process_pending(self, max_wait_seconds: float = 5.0) -> List[str]:
        """
        Process pending hypotheses that have been waiting too long.
        
        Args:
            max_wait_seconds: Maximum time to wait for votes
            
        Returns:
            List of resolved hypothesis IDs
        """
        resolved = []
        now = datetime.now()
        
        for hyp_id in list(self.pending_queue):
            hypothesis = self.hypotheses.get(hyp_id)
            if not hypothesis:
                continue
            
            age = (now - hypothesis.created_at).total_seconds()
            
            if age > max_wait_seconds:
                if hypothesis.status == DeliberationStatus.DEBATING:
                    self.resolve_debate(hyp_id)
                else:
                    self.force_resolution(hyp_id)
                resolved.append(hyp_id)
        
        return resolved
    
    def get_accepted_hypotheses(
        self,
        hypothesis_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Hypothesis]:
        """Get all accepted hypotheses."""
        results = [
            h for h in self.hypotheses.values()
            if h.status == DeliberationStatus.ACCEPTED
            and (h.final_confidence or 0) >= min_confidence
        ]
        
        if hypothesis_type:
            results = [h for h in results if h.hypothesis_type == hypothesis_type]
        
        return results
    
    def get_rejected_hypotheses(self) -> List[Hypothesis]:
        """Get all rejected hypotheses."""
        return [
            h for h in self.hypotheses.values()
            if h.status == DeliberationStatus.REJECTED
        ]
    
    def get_pending_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses still awaiting resolution."""
        return [
            h for h in self.hypotheses.values()
            if h.status in [DeliberationStatus.PENDING, DeliberationStatus.VOTING, DeliberationStatus.DEBATING]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deliberation statistics."""
        return {
            **self.stats,
            "pending": len(self.pending_queue),
            "total_hypotheses": len(self.hypotheses),
            "acceptance_rate": (
                self.stats["accepted"] / max(self.stats["consensus_reached"], 1)
            ),
        }


class AgentDeliberationMixin:
    """
    Mixin class providing deliberation capabilities to agents.
    
    Add this to BaseAgent to enable:
    - Hypothesis submission for voting
    - Voting on other agents' hypotheses
    - Debate participation
    - Consensus awareness
    """
    
    def __init__(self):
        # Will be set by the agent
        self.name: str = ""
        self.shared_memory: Optional[SharedMemory] = None
        self.message_bus: Optional[MessageBus] = None
        self._deliberation_coordinator: Optional[DeliberationCoordinator] = None
    
    def set_deliberation_coordinator(
        self,
        coordinator: DeliberationCoordinator,
    ) -> None:
        """Set the deliberation coordinator for this agent."""
        self._deliberation_coordinator = coordinator
    
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
            return None
        
        return self._deliberation_coordinator.submit_hypothesis(
            author=self.name,
            hypothesis_type=hypothesis_type,
            content=content,
            confidence=confidence,
            evidence=evidence,
            document_id=document_id,
        )
    
    def vote_on_hypothesis(
        self,
        hypothesis_id: str,
        vote_type: VoteType,
        confidence: float,
        rationale: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """
        Vote on another agent's hypothesis.
        
        Args:
            hypothesis_id: ID of hypothesis to vote on
            vote_type: Your vote
            confidence: Your confidence in this vote
            rationale: Reason for your vote
            evidence: Supporting evidence
        """
        if not self._deliberation_coordinator:
            return
        
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
        
        self._deliberation_coordinator.receive_debate_argument(
            hypothesis_id=hypothesis_id,
            agent=self.name,
            position=position,
            argument=argument,
            evidence=evidence,
        )
    
    def check_for_vote_requests(self) -> List[Dict[str, Any]]:
        """Check for pending vote requests from the coordinator."""
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
        """Check for pending debate requests from the coordinator."""
        if not self.message_bus:
            return []
        
        messages = self.message_bus.receive(self.name)
        debate_requests = [
            m.content for m in messages
            if m.comm_type == CommunicationType.REQUEST
            and m.content.get("action") == "debate"
        ]
        
        return debate_requests
    
    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[VoteType, float, str]:
        """
        Evaluate a hypothesis and determine your vote.
        
        Override this method in subclasses to implement
        agent-specific voting logic.
        
        Args:
            hypothesis_content: The hypothesis to evaluate
            hypothesis_type: Type of hypothesis
            context: Additional context
            
        Returns:
            Tuple of (vote_type, confidence, rationale)
        """
        # Default implementation - abstain
        return VoteType.ABSTAIN, 0.5, "No specific evaluation logic"
