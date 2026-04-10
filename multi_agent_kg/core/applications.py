"""
Application-layer factories on top of the governed knowledge graph.
"""

from __future__ import annotations

from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.incremental_enrichment import IncrementalEnricher
from multi_agent_kg.core.qa_orchestrator import QAOrchestrator


def create_qa_system(
    governed_kg: GovernedKnowledgeGraph,
    llm_config: LLMConfig,
    advanced: bool = True,
    **kwargs,
):
    if advanced:
        return AdvancedQAOrchestrator(
            governed_kg=governed_kg,
            llm_config=llm_config,
            **kwargs,
        )
    return QAOrchestrator(
        governed_kg=governed_kg,
        llm_config=llm_config,
        **kwargs,
    )


def create_enricher(
    governed_kg: GovernedKnowledgeGraph,
    llm_config: LLMConfig,
    **kwargs,
) -> IncrementalEnricher:
    return IncrementalEnricher(
        governed_kg=governed_kg,
        llm_config=llm_config,
        **kwargs,
    )


__all__ = ["create_qa_system", "create_enricher"]
