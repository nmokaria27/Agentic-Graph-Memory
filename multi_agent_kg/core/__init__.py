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

Deliberation:
- DeliberationCoordinator: Multi-agent voting and debate system
- VotingSession, DebateSession: Session tracking
- VoteType, DeliberationConfig: Configuration types

Orchestrator:
- DeliberativeOrchestrator: Full integrated multi-agent pipeline

Configuration:
- LLMConfig, RelationSchema: Configuration types

Core Data Structure:
- GovernedKnowledgeGraph: KG plus domain-governance layer
- GovernanceDecision: Audit record for governed updates

Application Layers:
- QAOrchestrator / AdvancedQAOrchestrator
- IncrementalEnricher
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
from multi_agent_kg.core.deliberation import (
    DeliberationCoordinator,
    Hypothesis,
    Vote,
    DebateArgument,
    VoteType,
    DeliberationStatus,
    VOTE_WEIGHTS,
)
from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator
from multi_agent_kg.core.adaptive_config import (
    DomainTaxonomy,
    DomainSchema,
    AdaptiveBatchCalculator,
    ThresholdAutoTuner,
    MODEL_SPECS,
)
from multi_agent_kg.core.kg_operations import (
    KGDiff,
    compute_diff,
    merge_kg,
    load_governed_kg,
    load_kg,
    save_governed_kg,
    save_kg,
    find_entity_matches,
    normalize_entity_name,
)
from multi_agent_kg.core.governed_kg import (
    GovernanceDecision,
    GovernedKnowledgeGraph,
)
from multi_agent_kg.core.applications import (
    create_enricher,
    create_qa_system,
)
from multi_agent_kg.core.incremental_enrichment import (
    IncrementalEnricher,
    ConflictResolver,
    GovernanceReviewBoard,
)
from multi_agent_kg.core.domain_experts import (
    Domain,
    TopicSubAgent,
    OrgChart,
    GovernanceAssignment,
    DomainBuilder,
    DomainExpertAgent,
    QAOrchestrator,
    find_paths,
    paths_to_text,
    neighbourhood,
)
from multi_agent_kg.core.advanced_qa import (
    AdvancedQAOrchestrator,
    ActiveExplorerExpert,
    CriticAgent,
    DebateArena,
    SessionMemory,
    ProvenanceChain,
    ProvenanceRecord,
)

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
    # Deliberation
    "DeliberationCoordinator",
    "Hypothesis",
    "Vote",
    "DebateArgument",
    "VoteType",
    "DeliberationStatus",
    "VOTE_WEIGHTS",
    # Orchestrator
    "DeliberativeOrchestrator",
    # Adaptive Configuration
    "DomainTaxonomy",
    "DomainSchema",
    "AdaptiveBatchCalculator",
    "ThresholdAutoTuner",
    "MODEL_SPECS",
    # KG Operations
    "KGDiff",
    "compute_diff",
    "merge_kg",
    "load_governed_kg",
    "load_kg",
    "save_governed_kg",
    "save_kg",
    "find_entity_matches",
    "GovernedKnowledgeGraph",
    "GovernanceDecision",
    "create_qa_system",
    "create_enricher",
    # Incremental Enrichment
    "IncrementalEnricher",
    "ConflictResolver",
    "GovernanceReviewBoard",
    # Domain Expert QA
    "Domain",
    "TopicSubAgent",
    "OrgChart",
    "GovernanceAssignment",
    "DomainBuilder",
    "DomainExpertAgent",
    "QAOrchestrator",
    # Multi-hop graph utilities
    "find_paths",
    "paths_to_text",
    "neighbourhood",
    # Advanced QA System
    "AdvancedQAOrchestrator",
    "ActiveExplorerExpert",
    "CriticAgent",
    "DebateArena",
    "SessionMemory",
    "ProvenanceChain",
    "ProvenanceRecord",
]
