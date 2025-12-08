"""
Multi-Agent Knowledge Graph Enrichment Framework
"""

__version__ = "0.1.0"
__author__ = "Multi-Agent KG Team"

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple, Entity
from multi_agent_kg.core.config import RelationType, RelationSchema

__all__ = [
    "KnowledgeGraph",
    "Triple",
    "Entity",
    "RelationType",
    "RelationSchema",
]
