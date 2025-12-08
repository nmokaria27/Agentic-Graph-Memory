"""
Message types for agent communication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class MessageType(str, Enum):
    """Types of messages that can be passed between agents."""

    RAW_TEXT = "raw_text"
    SEGMENTS = "segments"
    SUMMARIES = "summaries"
    ENTITIES = "entities"
    RELATIONS = "relations"
    TRIPLES = "triples"
    CONFLICTS = "conflicts"
    APPROVED_TRIPLES = "approved_triples"
    ERROR = "error"


@dataclass
class Message:
    """
    A message passed between agents in the system.

    Attributes:
        sender: Name of the agent sending the message
        receiver: Name of the agent receiving the message
        type: Type of message (from MessageType enum)
        payload: The actual data being passed
        meta: Optional metadata about the message
    """

    sender: str
    receiver: str
    type: MessageType
    payload: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Message(from={self.sender}, to={self.receiver}, type={self.type})"
