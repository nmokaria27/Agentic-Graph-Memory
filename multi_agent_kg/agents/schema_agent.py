"""
Schema alignment agent for validating entities and relations against the KG schema.
"""

from typing import Any, List, Optional
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.config import RelationSchema


class SchemaAgent(Agent):
    """
    Agent responsible for aligning extracted knowledge with the KG schema.

    Ensures:
    - All entities exist in the KG
    - All relations conform to the schema
    - Entity types are validated
    """

    def __init__(
        self,
        name: str = "SchemaAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        relation_schema: Optional[RelationSchema] = None,
    ):
        """
        Initialize the schema alignment agent.

        Args:
            name: Name of the agent
            knowledge_graph: Reference to the knowledge graph
            relation_schema: Schema defining allowed relation types
        """
        super().__init__(name, knowledge_graph)
        self.relation_schema = relation_schema or RelationSchema()

    def run(
        self,
        entities: List[Entity],
        triples: List[Triple],
        **kwargs: Any,
    ) -> List[Triple]:
        """
        Align entities and triples with the knowledge graph schema.

        Args:
            entities: List of entities to add to KG
            triples: List of triples to validate
            **kwargs: Additional arguments

        Returns:
            List of validated/cleaned triples
        """
        self.log(f"Aligning {len(entities)} entities and {len(triples)} triples with schema")

        # Ensure all entities are in the KG
        self._ensure_entities(entities)

        # Validate and clean triples
        cleaned_triples = self._validate_triples(triples)

        self.log(f"Schema alignment complete: {len(cleaned_triples)} valid triples")

        return cleaned_triples

    def _ensure_entities(self, entities: List[Entity]) -> None:
        """
        Ensure all entities exist in the knowledge graph.

        Args:
            entities: List of entities to add
        """
        if not self.knowledge_graph:
            self.log("No knowledge graph available", level="WARNING")
            return

        for entity in entities:
            self.knowledge_graph.add_entity(
                entity_id=entity.id,
                labels=entity.labels,
                entity_type=entity.type,
                metadata=entity.metadata,
            )

        self.log(f"Ensured {len(entities)} entities exist in KG")

    def _validate_triples(self, triples: List[Triple]) -> List[Triple]:
        """
        Validate triples against the relation schema.

        Args:
            triples: List of triples to validate

        Returns:
            List of valid triples
        """
        valid_triples = []
        invalid_count = 0

        for triple in triples:
            # Check if relation exists in schema
            if not self.relation_schema.has_relation_type(triple.relation):
                if self.relation_schema.allow_new_types:
                    # Allow unknown relations in open-world mode
                    self.log(
                        f"Unknown relation '{triple.relation}' allowed in open-world mode",
                        level="WARNING",
                    )
                    valid_triples.append(triple)
                else:
                    self.log(
                        f"Skipping triple with unknown relation: {triple.relation}",
                        level="WARNING",
                    )
                    invalid_count += 1
                    continue
            else:
                # Validate entity types if schema specifies constraints
                rel_type = self.relation_schema.get_relation_type(triple.relation)
                if rel_type and self.knowledge_graph:
                    subject_entity = self.knowledge_graph.entities.get(triple.subject)
                    object_entity = self.knowledge_graph.entities.get(triple.object)

                    subject_type = subject_entity.type if subject_entity else None
                    object_type = object_entity.type if object_entity else None

                    if not rel_type.validate_triple(subject_type, object_type):
                        self.log(
                            f"Triple {triple} violates schema constraints",
                            level="WARNING",
                        )
                        invalid_count += 1
                        continue

                valid_triples.append(triple)

        if invalid_count > 0:
            self.log(f"Filtered out {invalid_count} invalid triples")

        return valid_triples
