"""
Knowledge graph representation and operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import json


@dataclass
class Entity:
    """
    Represents an entity in the knowledge graph.

    Attributes:
        id: Unique identifier for the entity
        labels: Alternative names/labels for the entity
        type: Optional type/category of the entity
        metadata: Additional metadata about the entity
    """

    id: str
    labels: List[str] = field(default_factory=list)
    type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def __repr__(self) -> str:
        type_str = f":{self.type}" if self.type else ""
        return f"Entity({self.id}{type_str})"


@dataclass
class Triple:
    """
    Represents a relation triple (subject, relation, object) in the knowledge graph.

    Attributes:
        subject: Subject entity ID
        relation: Relation/predicate label
        object: Object entity ID
        confidence: Confidence score (0.0 to 1.0)
        source: Source of this triple (e.g., document ID, agent name)
        metadata: Additional metadata
    """

    subject: str
    relation: str
    object: str
    confidence: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject
            and self.relation == other.relation
            and self.object == other.object
        )

    def __repr__(self) -> str:
        conf_str = f" ({self.confidence:.2f})" if self.confidence else ""
        return f"({self.subject}) -[{self.relation}]-> ({self.object}){conf_str}"


@dataclass
class Conflict:
    """
    Represents a conflict between triples in the knowledge graph.

    Attributes:
        subject: Subject entity ID
        relation: Relation label
        existing_object: Object in existing triple
        new_object: Object in new/candidate triple
        existing_triple: The existing triple
        new_triple: The new/conflicting triple
    """

    subject: str
    relation: str
    existing_object: str
    new_object: str
    existing_triple: Triple
    new_triple: Triple

    def __repr__(self) -> str:
        return (
            f"Conflict({self.subject} -[{self.relation}]-> "
            f"{self.existing_object} vs {self.new_object})"
        )


class KnowledgeGraph:
    """
    A simple knowledge graph implementation.

    Stores entities (nodes) and triples (edges) with support for:
    - Adding entities and triples
    - Conflict detection
    - Querying and export
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self._triple_set: Set[Triple] = set()

    def add_entity(
        self,
        entity_id: str,
        labels: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """
        Add an entity to the knowledge graph.

        Args:
            entity_id: Unique identifier for the entity
            labels: Alternative names/labels
            entity_type: Type/category of the entity
            metadata: Additional metadata

        Returns:
            The Entity object (existing or newly created)
        """
        if entity_id in self.entities:
            # Entity exists, optionally merge labels/metadata
            existing = self.entities[entity_id]
            if labels:
                for label in labels:
                    if label not in existing.labels:
                        existing.labels.append(label)
            if entity_type and not existing.type:
                existing.type = entity_type
            if metadata:
                existing.metadata.update(metadata)
            return existing
        else:
            # Create new entity
            entity = Entity(
                id=entity_id,
                labels=labels or [],
                type=entity_type,
                metadata=metadata or {},
            )
            self.entities[entity_id] = entity
            return entity

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Triple]:
        """
        Add a triple to the knowledge graph.

        Args:
            subject: Subject entity ID
            relation: Relation/predicate label
            obj: Object entity ID
            confidence: Confidence score
            source: Source of the triple
            metadata: Additional metadata

        Returns:
            The Triple object if added, None if duplicate
        """
        triple = Triple(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )

        # Check for duplicates
        if triple in self._triple_set:
            return None

        self.triples.append(triple)
        self._triple_set.add(triple)

        # Do NOT auto-create phantom entities.  The KnowledgeOrganizer is
        # responsible for ensuring that subject/object IDs exist before
        # calling add_triple.  If they somehow don't exist we still allow
        # the triple but log the gap — callers can audit via
        # ``get_orphan_triples()``.
        return triple

    def get_orphan_triples(self) -> List[Triple]:
        """Return triples whose subject or object has no matching entity."""
        return [
            t for t in self.triples
            if t.subject not in self.entities or t.object not in self.entities
        ]

    def find_conflicts(self, candidate_triples: List[Triple]) -> List[Conflict]:
        """
        Find conflicts between candidate triples and existing triples.

        A conflict is defined as: same subject and relation but different object.

        Args:
            candidate_triples: List of triples to check for conflicts

        Returns:
            List of Conflict objects
        """
        conflicts: List[Conflict] = []

        # Build a map of (subject, relation) -> objects
        existing_map: Dict[tuple, List[Triple]] = {}
        for triple in self.triples:
            key = (triple.subject, triple.relation)
            if key not in existing_map:
                existing_map[key] = []
            existing_map[key].append(triple)

        # Check candidates for conflicts
        for candidate in candidate_triples:
            key = (candidate.subject, candidate.relation)
            if key in existing_map:
                for existing in existing_map[key]:
                    if existing.object != candidate.object:
                        conflicts.append(
                            Conflict(
                                subject=candidate.subject,
                                relation=candidate.relation,
                                existing_object=existing.object,
                                new_object=candidate.object,
                                existing_triple=existing,
                                new_triple=candidate,
                            )
                        )

        return conflicts

    def get_triples_by_subject(self, subject: str) -> List[Triple]:
        """Get all triples with the given subject."""
        return [t for t in self.triples if t.subject == subject]

    def get_triples_by_relation(self, relation: str) -> List[Triple]:
        """Get all triples with the given relation."""
        return [t for t in self.triples if t.relation == relation]

    def get_triples_by_object(self, obj: str) -> List[Triple]:
        """Get all triples with the given object."""
        return [t for t in self.triples if t.object == obj]

    def print_graph(self) -> None:
        """Print a human-readable representation of the knowledge graph."""
        print("=" * 60)
        print("KNOWLEDGE GRAPH")
        print("=" * 60)

        print(f"\nEntities ({len(self.entities)}):")
        print("-" * 60)
        for entity in self.entities.values():
            labels_str = f" [{', '.join(entity.labels)}]" if entity.labels else ""
            type_str = f" : {entity.type}" if entity.type else ""
            print(f"  • {entity.id}{type_str}{labels_str}")

        print(f"\nTriples ({len(self.triples)}):")
        print("-" * 60)
        for triple in self.triples:
            conf_str = f" [confidence: {triple.confidence:.2f}]" if triple.confidence else ""
            source_str = f" (source: {triple.source})" if triple.source else ""
            print(f"  • {triple}{conf_str}{source_str}")

        print("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Export the knowledge graph as a dictionary."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "labels": e.labels,
                    "type": e.type,
                    "metadata": e.metadata,
                }
                for e in self.entities.values()
            ],
            "triples": [
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                    "confidence": t.confidence,
                    "source": t.source,
                    "metadata": t.metadata,
                }
                for t in self.triples
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export the knowledge graph as JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruct a knowledge graph from serialized data."""
        kg = cls()
        entities_list = data.get("entities", [])
        triples_list = data.get("triples", [])

        for entity_data in entities_list:
            kg.add_entity(
                entity_id=entity_data["id"],
                labels=entity_data.get("labels", []),
                entity_type=entity_data.get("type"),
                metadata=entity_data.get("metadata", {}),
            )

        for triple_data in triples_list:
            subject = triple_data.get("subject", "")
            relation = triple_data.get("relation", "")
            obj = triple_data.get("object", "")
            if subject and relation and obj:
                kg.add_triple(
                    subject=subject,
                    relation=relation,
                    obj=obj,
                    confidence=triple_data.get("confidence"),
                    source=triple_data.get("source"),
                    metadata=triple_data.get("metadata", {}),
                )

        return kg

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        relation_counts: Dict[str, int] = {}
        for triple in self.triples:
            relation_counts[triple.relation] = relation_counts.get(triple.relation, 0) + 1

        entity_type_counts: Dict[str, int] = {}
        for entity in self.entities.values():
            if entity.type:
                entity_type_counts[entity.type] = entity_type_counts.get(entity.type, 0) + 1

        return {
            "num_entities": len(self.entities),
            "num_triples": len(self.triples),
            "relation_counts": relation_counts,
            "entity_type_counts": entity_type_counts,
        }
