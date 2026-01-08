"""
Agent Communication Protocol for Multi-Agent Collaboration.

Implements:
- Direct agent-to-agent messaging
- Broadcast communication
- Request-response patterns
- Collaborative refinement loops
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import uuid


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class CommunicationType(str, Enum):
    """Types of inter-agent communication."""
    INFORM = "inform"           # Sharing information
    REQUEST = "request"         # Asking for something
    RESPONSE = "response"       # Responding to a request
    PROPOSE = "propose"         # Proposing a hypothesis
    ACCEPT = "accept"           # Accepting a proposal
    REJECT = "reject"           # Rejecting a proposal
    REFINE = "refine"           # Suggesting refinement
    DELEGATE = "delegate"       # Delegating a task
    FEEDBACK = "feedback"       # Providing feedback


@dataclass
class AgentMessage:
    """
    Message passed between agents.
    """
    id: str
    sender: str
    receiver: str                         # Can be agent name or "broadcast"
    comm_type: CommunicationType
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    in_reply_to: Optional[str] = None     # Reference to previous message
    requires_response: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "comm_type": self.comm_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "in_reply_to": self.in_reply_to,
            "requires_response": self.requires_response,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MessageBus:
    """
    Central message bus for agent communication.
    
    Supports:
    - Point-to-point messaging
    - Broadcast messaging
    - Topic-based pub/sub
    - Message history
    """

    def __init__(self, debug_logger=None):
        self.messages: List[AgentMessage] = []
        self.pending_responses: Dict[str, AgentMessage] = {}
        self.subscribers: Dict[str, List[Callable]] = {}  # topic -> callbacks
        self.agent_inboxes: Dict[str, List[AgentMessage]] = {}
        self.debug_logger = debug_logger

    def send(
        self,
        sender: str,
        receiver: str,
        comm_type: CommunicationType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        in_reply_to: Optional[str] = None,
        requires_response: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            sender: Sending agent name
            receiver: Receiving agent name or "broadcast"
            comm_type: Type of communication
            content: Message content
            priority: Message priority
            in_reply_to: ID of message being replied to
            requires_response: Whether a response is expected
            metadata: Additional metadata
            
        Returns:
            Message ID
        """
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        message = AgentMessage(
            id=msg_id,
            sender=sender,
            receiver=receiver,
            comm_type=comm_type,
            content=content,
            priority=priority,
            in_reply_to=in_reply_to,
            requires_response=requires_response,
            metadata=metadata or {},
        )
        
        self.messages.append(message)
        
        # Debug log the message
        if self.debug_logger:
            self.debug_logger.log_agent_message(
                sender=sender,
                receiver=receiver,
                message_type=comm_type.value,
                content=content,
                priority=priority.value
            )
        
        if requires_response:
            self.pending_responses[msg_id] = message
        
        # Deliver to inbox
        if receiver == "broadcast":
            for inbox in self.agent_inboxes.values():
                inbox.append(message)
        else:
            if receiver not in self.agent_inboxes:
                self.agent_inboxes[receiver] = []
            self.agent_inboxes[receiver].append(message)
        
        # Notify topic subscribers
        topic = content.get("topic")
        if topic and topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception:
                    pass
        
        return msg_id

    def receive(
        self,
        agent_name: str,
        comm_type: Optional[CommunicationType] = None,
        priority: Optional[MessagePriority] = None,
    ) -> List[AgentMessage]:
        """
        Receive messages for an agent.
        
        Args:
            agent_name: Agent checking for messages
            comm_type: Filter by communication type
            priority: Filter by priority
            
        Returns:
            List of messages
        """
        if agent_name not in self.agent_inboxes:
            return []
        
        messages = self.agent_inboxes[agent_name]
        
        if comm_type:
            messages = [m for m in messages if m.comm_type == comm_type]
        
        if priority:
            messages = [m for m in messages if m.priority == priority]
        
        return sorted(messages, key=lambda m: (m.priority.value, m.timestamp))

    def clear_inbox(self, agent_name: str) -> None:
        """Clear an agent's inbox."""
        if agent_name in self.agent_inboxes:
            self.agent_inboxes[agent_name] = []

    def reply(
        self,
        original_msg_id: str,
        sender: str,
        comm_type: CommunicationType,
        content: Dict[str, Any],
    ) -> str:
        """
        Reply to a message.
        
        Args:
            original_msg_id: ID of message being replied to
            sender: Replying agent
            comm_type: Type of reply
            content: Reply content
            
        Returns:
            Reply message ID
        """
        # Find original message
        original = next((m for m in self.messages if m.id == original_msg_id), None)
        if not original:
            raise ValueError(f"Original message {original_msg_id} not found")
        
        reply_id = self.send(
            sender=sender,
            receiver=original.sender,
            comm_type=comm_type,
            content=content,
            in_reply_to=original_msg_id,
        )
        
        # Remove from pending if response was required
        if original_msg_id in self.pending_responses:
            del self.pending_responses[original_msg_id]
        
        return reply_id

    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic] = [c for c in self.subscribers[topic] if c != callback]

    def get_conversation(self, message_id: str) -> List[AgentMessage]:
        """Get the full conversation thread for a message."""
        thread = []
        current_id = message_id
        
        # Follow replies backward
        while current_id:
            msg = next((m for m in self.messages if m.id == current_id), None)
            if msg:
                thread.insert(0, msg)
                current_id = msg.in_reply_to
            else:
                break
        
        # Follow replies forward
        def find_replies(msg_id: str):
            return [m for m in self.messages if m.in_reply_to == msg_id]
        
        to_process = [message_id]
        while to_process:
            current = to_process.pop(0)
            replies = find_replies(current)
            for reply in replies:
                if reply not in thread:
                    thread.append(reply)
                    to_process.append(reply.id)
        
        return sorted(thread, key=lambda m: m.timestamp)

    def get_pending_requests(self) -> List[AgentMessage]:
        """Get all pending requests awaiting responses."""
        return list(self.pending_responses.values())

    def get_history(
        self,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        comm_type: Optional[CommunicationType] = None,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """Get message history with filters."""
        messages = self.messages
        
        if sender:
            messages = [m for m in messages if m.sender == sender]
        
        if receiver:
            messages = [m for m in messages if m.receiver == receiver]
        
        if comm_type:
            messages = [m for m in messages if m.comm_type == comm_type]
        
        return sorted(messages, key=lambda m: m.timestamp, reverse=True)[:limit]


class CollaborationProtocol:
    """
    High-level collaboration patterns for agents.
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus

    def request_refinement(
        self,
        requester: str,
        target_agent: str,
        item: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Request another agent to refine an extraction.
        
        Args:
            requester: Agent requesting refinement
            target_agent: Agent to do the refinement
            item: Item to be refined (entity, triple, etc.)
            context: Additional context
            
        Returns:
            Request message ID
        """
        return self.bus.send(
            sender=requester,
            receiver=target_agent,
            comm_type=CommunicationType.REQUEST,
            content={
                "action": "refine",
                "item": item,
                "context": context or {},
            },
            requires_response=True,
        )

    def propose_hypothesis(
        self,
        proposer: str,
        hypothesis: Dict[str, Any],
        confidence: float,
        evidence: Optional[List[str]] = None,
    ) -> str:
        """
        Propose a hypothesis for other agents to vote on.
        
        Args:
            proposer: Agent making the proposal
            hypothesis: The hypothesis (e.g., a triple)
            confidence: Proposer's confidence
            evidence: Supporting evidence
            
        Returns:
            Proposal message ID
        """
        return self.bus.send(
            sender=proposer,
            receiver="broadcast",
            comm_type=CommunicationType.PROPOSE,
            content={
                "hypothesis": hypothesis,
                "confidence": confidence,
                "evidence": evidence or [],
            },
            requires_response=True,
        )

    def vote_on_proposal(
        self,
        voter: str,
        proposal_id: str,
        accept: bool,
        confidence: float,
        rationale: str,
    ) -> str:
        """
        Vote on a proposal.
        
        Args:
            voter: Voting agent
            proposal_id: ID of the proposal
            accept: Whether to accept
            confidence: Voter's confidence
            rationale: Reason for the vote
            
        Returns:
            Vote message ID
        """
        comm_type = CommunicationType.ACCEPT if accept else CommunicationType.REJECT
        
        return self.bus.reply(
            original_msg_id=proposal_id,
            sender=voter,
            comm_type=comm_type,
            content={
                "vote": "accept" if accept else "reject",
                "confidence": confidence,
                "rationale": rationale,
            },
        )

    def delegate_task(
        self,
        delegator: str,
        delegate: str,
        task: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Delegate a task to another agent.
        
        Args:
            delegator: Agent delegating the task
            delegate: Agent to perform the task
            task: Task specification
            priority: Task priority
            
        Returns:
            Delegation message ID
        """
        return self.bus.send(
            sender=delegator,
            receiver=delegate,
            comm_type=CommunicationType.DELEGATE,
            content={"task": task},
            priority=priority,
            requires_response=True,
        )

    def provide_feedback(
        self,
        provider: str,
        recipient: str,
        original_output: Dict[str, Any],
        feedback: Dict[str, Any],
        suggested_corrections: Optional[List[Dict]] = None,
    ) -> str:
        """
        Provide feedback on another agent's output.
        
        Args:
            provider: Agent providing feedback
            recipient: Agent receiving feedback
            original_output: The output being critiqued
            feedback: Feedback content
            suggested_corrections: Suggested fixes
            
        Returns:
            Feedback message ID
        """
        return self.bus.send(
            sender=provider,
            receiver=recipient,
            comm_type=CommunicationType.FEEDBACK,
            content={
                "original_output": original_output,
                "feedback": feedback,
                "suggested_corrections": suggested_corrections or [],
            },
        )
