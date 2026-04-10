"""
Backward-compatible re-export shim for the refactored governance and QA modules.
"""

from multi_agent_kg.core.domain_builder import DomainBuilder
from multi_agent_kg.core.governance import (
    Domain,
    GovernanceAssignment,
    OrgChart,
    TopicSubAgent,
)
from multi_agent_kg.core.graph_traversal import (
    find_paths,
    neighbourhood,
    paths_to_text,
)
from multi_agent_kg.core.qa_orchestrator import (
    DomainExpertAgent,
    FallbackGraphExpert,
    QAOrchestrator,
)
from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json

__all__ = [
    "TopicSubAgent",
    "Domain",
    "OrgChart",
    "GovernanceAssignment",
    "find_paths",
    "paths_to_text",
    "neighbourhood",
    "DomainBuilder",
    "DomainExpertAgent",
    "FallbackGraphExpert",
    "QAOrchestrator",
    "chat_completion",
    "chat_completion_json",
]
