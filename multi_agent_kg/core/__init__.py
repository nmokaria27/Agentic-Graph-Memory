"""Core module initialization."""

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple, Entity, Conflict
from multi_agent_kg.core.messages import Message, MessageType
from multi_agent_kg.core.config import RelationType, RelationSchema, LLMConfig
from multi_agent_kg.core.memory import SharedMemory, MemoryType, MemoryEntry
from multi_agent_kg.core.communication import MessageBus, AgentMessage, CollaborationProtocol

__all__ = [
    "KnowledgeGraph",
    "Triple",
    "Entity",
    "Conflict",
    "Message",
    "MessageType",
    "RelationType",
    "RelationSchema",
    "LLMConfig",
    "SharedMemory",
    "MemoryType",
    "MemoryEntry",
    "MessageBus",
    "AgentMessage",
    "CollaborationProtocol",
]
