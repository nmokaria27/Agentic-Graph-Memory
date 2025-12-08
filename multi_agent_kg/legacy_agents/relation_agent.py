"""
Relation extraction agent with configurable schema support.
"""

from typing import Any, List, Optional
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.config import LLMConfig, RelationSchema, RelationType
from multi_agent_kg.llm.openai_client import chat_completion_json
import json


class RelationAgent(Agent):
    """
    Agent responsible for extracting relations between entities using LLM.

    Supports both schema-constrained and open-world relation extraction.
    """

    def __init__(
        self,
        name: str = "RelationAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        relation_schema: Optional[RelationSchema] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the relation extraction agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
            relation_schema: Schema defining allowed relation types
            llm_config: Configuration for LLM calls
        """
        super().__init__(name, knowledge_graph)
        self.relation_schema = relation_schema or RelationSchema()
        self.llm_config = llm_config or LLMConfig(temperature=0.1)

    def run(
        self,
        text_chunks: List[str],
        entities: List[Entity],
        open_world: bool = False,
        **kwargs: Any,
    ) -> List[Triple]:
        """
        Extract relations from text given a list of entities.

        Args:
            text_chunks: List of text segments
            entities: List of known entities
            open_world: If True, allow discovery of new relation types
            **kwargs: Additional arguments

        Returns:
            List of Triple objects
        """
        self.log(
            f"Extracting relations from {len(text_chunks)} chunks "
            f"with {len(entities)} entities (open_world={open_world})"
        )

        # Combine text chunks
        combined_text = "\n\n".join(text_chunks)

        # Extract relations
        triples = self._extract_relations(combined_text, entities, open_world)

        self.log(f"Extracted {len(triples)} relation triples")

        return triples

    def _extract_relations(
        self, text: str, entities: List[Entity], open_world: bool
    ) -> List[Triple]:
        """
        Extract relations from text.

        Args:
            text: Text to process
            entities: List of entities
            open_world: Whether to allow new relation types

        Returns:
            List of Triple objects
        """
        # Prepare entity information
        entity_info = self._format_entities(entities)

        # Prepare relation schema information
        if open_world:
            schema_info = (
                "You may use relations from the schema below OR introduce new relation types "
                "if they better describe the relationships in the text.\n\n"
                + self.relation_schema.get_schema_description()
            )
        else:
            schema_info = (
                "You must ONLY use the following relation types:\n\n"
                + self.relation_schema.get_schema_description()
            )

        system_msg = {
            "role": "system",
            "content": (
                "You are a relation extraction system. "
                "Extract relationships between entities from the given text. "
                "Be precise and only extract relations that are clearly stated or strongly implied."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"{schema_info}\n\n"
                f"Entities:\n{entity_info}\n\n"
                f"Text:\n{text}\n\n"
                f"Extract all relationships between the entities.\n"
                f"Return a JSON object with a 'relations' key containing an array of objects.\n"
                f"Each object should have:\n"
                f"- subject: entity name (must be from the entity list)\n"
                f"- relation: relation type\n"
                f"- object: entity name (must be from the entity list)\n"
                f"- confidence: confidence score from 0.0 to 1.0\n"
                f"- rationale: brief explanation (1 sentence)\n"
            ),
        }

        if open_world:
            user_msg["content"] += (
                f"\n- relation_definition: (only if introducing a new relation type) "
                f"a brief definition of the relation\n"
            )

        try:
            response = chat_completion_json(
                messages=[system_msg, user_msg],
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens or 2000,
            )

            # Parse response
            triples = []
            relations = response.get("relations", [])

            for item in relations:
                subject = item.get("subject", "").strip()
                relation = item.get("relation", "").strip()
                obj = item.get("object", "").strip()
                confidence = item.get("confidence", 0.8)
                rationale = item.get("rationale", "")

                if not (subject and relation and obj):
                    continue

                # Validate entities exist
                entity_ids = {e.id for e in entities}
                if subject not in entity_ids or obj not in entity_ids:
                    self.log(
                        f"Skipping triple with unknown entities: {subject} -> {obj}",
                        level="WARNING",
                    )
                    continue

                # Handle new relation types in open world mode
                if open_world and not self.relation_schema.has_relation_type(relation):
                    definition = item.get("relation_definition", "")
                    if definition:
                        new_rel_type = RelationType(name=relation, description=definition)
                        self.relation_schema.add_relation_type(new_rel_type)
                        self.log(f"Discovered new relation type: {relation}")

                triple = Triple(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    confidence=confidence,
                    source=self.name,
                    metadata={"rationale": rationale},
                )
                triples.append(triple)

            return triples

        except Exception as e:
            self.log(f"Relation extraction failed: {e}", level="ERROR")
            return []

    def _format_entities(self, entities: List[Entity]) -> str:
        """
        Format entities for the LLM prompt.

        Args:
            entities: List of entities

        Returns:
            Formatted string
        """
        lines = []
        for entity in entities:
            type_str = f" (type: {entity.type})" if entity.type else ""
            aliases_str = f" [aliases: {', '.join(entity.labels)}]" if entity.labels else ""
            lines.append(f"- {entity.id}{type_str}{aliases_str}")
        return "\n".join(lines)
