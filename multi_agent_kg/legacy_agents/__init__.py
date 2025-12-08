"""
Legacy Agents Module.

⚠️ DEPRECATED: These agents are from the original implementation and are
kept for reference only. They are NOT importable without modifications.

For new projects, use the integrated agents:
    from multi_agent_kg.agents import (
        DocumentProcessor,
        DomainClassifier,
        EntityExtractor,
        RelationExtractor,
        EvidenceLinker,
        ExtractionValidator,
        ExtractionVerificationAgent,
        KnowledgeOrganizer,
    )

Legacy Agents (reference only - files preserved):
- base_agent.py: Original base agent class
- ingestion_agent.py: Document loading
- segmenter_agent.py: Text segmentation
- summarizer_agent.py: Text summarization
- entity_agent.py: Entity extraction
- relation_agent.py: Relation extraction (schema-based)
- open_world_relation_agent.py: Open-world relation extraction
- schema_agent.py: Schema alignment
- conflict_agent.py: Conflict detection
- verifier_agent.py: Triple verification

Legacy Orchestrators (reference only):
- orchestrator.py: Basic pipeline orchestrator
- advanced_orchestrator.py: Orchestrator with memory (partial integration)

Legacy Examples (reference only):
- run_pipeline.py: Basic pipeline example
- advanced_pipeline.py: Advanced pipeline example
"""

import warnings

warnings.warn(
    "The legacy_agents module contains deprecated code kept for reference. "
    "Use multi_agent_kg.agents for the integrated agent system.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
