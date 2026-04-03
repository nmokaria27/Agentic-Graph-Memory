"""
Deep Agent integration layer for the multi-agent KG pipeline.

Provides ``DeepAgentOrchestrator`` which wraps the existing
``DeliberativeOrchestrator`` pipeline into a LangChain Deep Agents-
compatible interface.  When the ``deepagents`` library is not installed
the orchestrator falls back transparently to the standard pipeline.
"""

from multi_agent_kg.deep_agent.agent import (
    DeepAgentOrchestrator,
    DEEP_AGENTS_AVAILABLE,
)

__all__ = [
    "DeepAgentOrchestrator",
    "DEEP_AGENTS_AVAILABLE",
]
