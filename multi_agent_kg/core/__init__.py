"""
Multi-Agent Knowledge Graph Core Module.

This module contains the core infrastructure:

Knowledge Graph:
- KnowledgeGraph: The main knowledge graph data structure
- Triple, Entity, Conflict: Core data types

Memory System:
- SharedMemory: Episodic, semantic, working memory + blackboard
- MemoryType, MemoryEntry: Memory data types

Communication:
- MessageBus: Inter-agent communication
- AgentMessage, CollaborationProtocol: Communication types

Orchestrator:
- DeliberativeOrchestrator: Full integrated multi-agent pipeline

Configuration:
- LLMConfig, RelationSchema: Configuration types
"""

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple, Entity, Conflict
from multi_agent_kg.core.messages import Message, MessageType
from multi_agent_kg.core.config import RelationType, RelationSchema, LLMConfig
from multi_agent_kg.core.memory import SharedMemory, MemoryType, MemoryEntry
from multi_agent_kg.core.communication import (
    MessageBus, 
    AgentMessage, 
    CollaborationProtocol,
    CommunicationType,
    MessagePriority,
)
from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator

__all__ = [
    # Knowledge Graph
    "KnowledgeGraph",
    "Triple",
    "Entity",
    "Conflict",
    # Messages
    "Message",
    "MessageType",
    # Config
    "RelationType",
    "RelationSchema",
    "LLMConfig",
    # Memory
    "SharedMemory",
    "MemoryType",
    "MemoryEntry",
    # Communication
    "MessageBus",
    "AgentMessage",
    "CollaborationProtocol",
    "CommunicationType",
    "MessagePriority",
    # Orchestrator
    "DeliberativeOrchestrator",
]
