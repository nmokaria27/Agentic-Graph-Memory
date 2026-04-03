"""
Multi-Agent Knowledge Graph Agents Package.

This package contains the integrated agents for knowledge graph enrichment.

Worker Agents (extraction):
- DocumentProcessor: Document ingestion and segmentation
- DomainClassifier: Domain classification for tailored extraction
- EntityExtractor: Multi-stage entity extraction with self-consistency
- RelationExtractor: RHF-style relation extraction with open-world support
- EvidenceLinker: Evidence linking and cross-referencing

Coordinator Agents (validation):
- ExtractionValidator: Validation and iterative refinement
- ExtractionVerificationAgent: Final verification before KG integration
- KnowledgeOrganizer: KG integration and maintenance

Base Classes:
- BaseAgent: Enhanced base with memory and communication integration
- AgentRole, ModelTier, AgentContext, ExtractionResult: Core types

For legacy agents, use: from multi_agent_kg.legacy_agents import ...
"""

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    ModelTier,
    AgentContext,
    ExtractionResult,
)

from multi_agent_kg.agents.document_processor import DocumentProcessor
from multi_agent_kg.agents.domain_classifier import DomainClassifier
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.agents.evidence_linker import EvidenceLinker

from multi_agent_kg.agents.extraction_validator import ExtractionValidator
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.agents.entity_resolver import EntityResolver
from multi_agent_kg.agents.critic_agent import CriticAgent
from multi_agent_kg.agents.corrector_agent import CorrectorAgent
from multi_agent_kg.agents.triplex_extractor import TriplexExtractor
from multi_agent_kg.agents.schema_aligner import SchemaAligner
from multi_agent_kg.agents.orphan_linker import OrphanLinker

__all__ = [
    # Base
    "BaseAgent",
    "AgentRole",
    "ModelTier",
    "AgentContext",
    "ExtractionResult",
    # Workers
    "DocumentProcessor",
    "DomainClassifier",
    "EntityExtractor",
    "RelationExtractor",
    "EvidenceLinker",
    "EntityResolver",
    "CriticAgent",
    "CorrectorAgent",
    "TriplexExtractor",
    "SchemaAligner",
    # Coordinators
    "ExtractionValidator",
    "ExtractionVerificationAgent",
    "KnowledgeOrganizer",
    "OrphanLinker",
]
