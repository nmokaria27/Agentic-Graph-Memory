"""
Configuration classes for the multi-agent knowledge graph system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RelationType:
    """
    Defines a type of relation that can exist in the knowledge graph.

    Attributes:
        name: The name/label of the relation (e.g., "treats", "causes")
        description: A clear description of what this relation means
        allowed_subject_types: Optional list of entity types that can be subjects
        allowed_object_types: Optional list of entity types that can be objects
    """

    name: str
    description: str
    allowed_subject_types: Optional[List[str]] = None
    allowed_object_types: Optional[List[str]] = None

    def __repr__(self) -> str:
        return f"RelationType({self.name})"

    def validate_triple(self, subject_type: Optional[str], object_type: Optional[str]) -> bool:
        """
        Check if a triple with given subject/object types is valid for this relation.

        Args:
            subject_type: Type of the subject entity
            object_type: Type of the object entity

        Returns:
            True if the triple is valid, False otherwise
        """
        if self.allowed_subject_types and subject_type:
            if subject_type not in self.allowed_subject_types:
                return False

        if self.allowed_object_types and object_type:
            if object_type not in self.allowed_object_types:
                return False

        return True


@dataclass
class RelationSchema:
    """
    Collection of relation types that define the schema for the knowledge graph.

    Attributes:
        types: List of allowed relation types
        allow_new_types: Whether to allow discovery of new relation types
    """

    types: List[RelationType] = field(default_factory=list)
    allow_new_types: bool = False

    def get_relation_type(self, name: str) -> Optional[RelationType]:
        """Get a relation type by name."""
        for rel_type in self.types:
            if rel_type.name == name:
                return rel_type
        return None

    def has_relation_type(self, name: str) -> bool:
        """Check if a relation type exists in the schema."""
        return self.get_relation_type(name) is not None

    def add_relation_type(self, relation_type: RelationType) -> None:
        """Add a new relation type to the schema."""
        if not self.has_relation_type(relation_type.name):
            self.types.append(relation_type)

    def get_schema_description(self) -> str:
        """Get a formatted description of all relation types for LLM prompts."""
        lines = ["Available relation types:"]
        for rel_type in self.types:
            lines.append(f"  - {rel_type.name}: {rel_type.description}")
            if rel_type.allowed_subject_types:
                lines.append(f"    Subject types: {', '.join(rel_type.allowed_subject_types)}")
            if rel_type.allowed_object_types:
                lines.append(f"    Object types: {', '.join(rel_type.allowed_object_types)}")
        return "\n".join(lines)


@dataclass
class LLMConfig:
    """
    Configuration for LLM calls.

    Attributes:
        model: The model to use (e.g., "gemma3:27b", "qwen3:8b")
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
    """

    model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b"))
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API calls."""
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens:
            config["max_tokens"] = self.max_tokens
        return config
