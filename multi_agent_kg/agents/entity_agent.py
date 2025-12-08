"""
Entity extraction agent using LLM.
"""

from typing import Any, List, Optional, Dict
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion_json


class EntityAgent(Agent):
    """
    Agent responsible for extracting entities from text using LLM.
    """

    def __init__(
        self,
        name: str = "EntityAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the entity extraction agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
            llm_config: Configuration for LLM calls
        """
        super().__init__(name, knowledge_graph)
        self.llm_config = llm_config or LLMConfig(temperature=0.1)

    def run(
        self,
        text_chunks: List[str],
        **kwargs: Any,
    ) -> List[Entity]:
        """
        Extract entities from a list of text chunks.

        Args:
            text_chunks: List of text segments to process
            **kwargs: Additional arguments

        Returns:
            List of Entity objects
        """
        self.log(f"Extracting entities from {len(text_chunks)} text chunks")

        all_entities: Dict[str, Entity] = {}

        for i, chunk in enumerate(text_chunks):
            self.log(f"Processing chunk {i + 1}/{len(text_chunks)}")
            entities = self._extract_from_chunk(chunk)

            # Merge entities (deduplicate by ID)
            for entity in entities:
                if entity.id in all_entities:
                    # Merge labels
                    existing = all_entities[entity.id]
                    for label in entity.labels:
                        if label not in existing.labels:
                            existing.labels.append(label)
                    # Update type if not set
                    if not existing.type and entity.type:
                        existing.type = entity.type
                else:
                    all_entities[entity.id] = entity

        entity_list = list(all_entities.values())
        self.log(f"Extracted {len(entity_list)} unique entities")

        return entity_list

    def _extract_from_chunk(self, text: str) -> List[Entity]:
        """
        Extract entities from a single text chunk.

        Args:
            text: Text to process

        Returns:
            List of Entity objects
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are an entity extraction system. "
                "Extract all important entities (people, organizations, locations, "
                "concepts, diseases, drugs, etc.) from the text. "
                "For each entity, provide a normalized name and optional type."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Extract all entities from the following text.\n\n"
                f"Text:\n{text}\n\n"
                f"Return a JSON array of objects with keys:\n"
                f"- name: the entity name (normalized)\n"
                f"- type: the entity type (e.g., Person, Organization, Location, Disease, Drug, Concept, etc.)\n"
                f"- aliases: optional array of alternative names\n\n"
                f"Example output:\n"
                f'{{"entities": [{{"name": "Aspirin", "type": "Drug", "aliases": ["acetylsalicylic acid"]}}]}}'
            ),
        }

        try:
            response = chat_completion_json(
                messages=[system_msg, user_msg],
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )

            # Parse response
            entities = []
            entity_list = response.get("entities", [])

            for item in entity_list:
                name = item.get("name", "").strip()
                if not name:
                    continue

                entity_type = item.get("type")
                aliases = item.get("aliases", [])

                # Use name as ID (could be improved with deduplication)
                entity = Entity(
                    id=name,
                    labels=aliases if aliases else [],
                    type=entity_type,
                )
                entities.append(entity)

            return entities

        except Exception as e:
            self.log(f"Entity extraction failed: {e}", level="ERROR")
            return []
