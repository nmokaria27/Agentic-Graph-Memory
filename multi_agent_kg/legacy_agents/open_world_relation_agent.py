"""
Open-World Relation Extraction Agent.

Discovers relations without predefined schemas:
- Extracts any meaningful relationship
- Proposes new relation types
- Learns from context and previous extractions
- Collaborates with other agents for refinement
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CommunicationType, CollaborationProtocol
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion_json


@dataclass
class DiscoveredRelation:
    """A newly discovered relation type."""
    name: str
    definition: str
    examples: List[Tuple[str, str, str]] = field(default_factory=list)  # (subj, rel, obj)
    frequency: int = 1
    confidence: float = 0.5
    source_documents: List[str] = field(default_factory=list)


class OpenWorldRelationAgent(Agent):
    """
    Agent that extracts relations without predefined schemas.
    
    Features:
    - Discovers new relation types from text
    - Learns relation patterns across documents
    - Uses shared memory for context
    - Collaborates with other agents
    """

    def __init__(
        self,
        name: str = "OpenWorldRelationAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        super().__init__(name, knowledge_graph)
        self.shared_memory = shared_memory
        self.message_bus = message_bus
        self.llm_config = llm_config or LLMConfig(temperature=0.3)
        
        # Track discovered relations
        self.discovered_relations: Dict[str, DiscoveredRelation] = {}
        
        # Collaboration protocol
        if message_bus:
            self.collab = CollaborationProtocol(message_bus)
        else:
            self.collab = None

    def run(
        self,
        text_chunks: List[str],
        entities: List[Entity],
        document_id: Optional[str] = None,
        use_context: bool = True,
        propose_to_agents: bool = True,
        **kwargs: Any,
    ) -> List[Triple]:
        """
        Extract relations in open-world mode.
        
        Args:
            text_chunks: Text segments to process
            entities: Extracted entities
            document_id: ID for tracking document provenance
            use_context: Whether to use shared memory context
            propose_to_agents: Whether to propose triples to other agents
            
        Returns:
            List of extracted triples
        """
        self.log(f"Open-world extraction from {len(text_chunks)} chunks, {len(entities)} entities")
        
        # Get context from memory if available
        context = self._get_relevant_context(entities) if use_context else {}
        
        # Get previously discovered relations for guidance
        known_relations = self._get_known_relations()
        
        # Extract relations
        all_triples = []
        for i, chunk in enumerate(text_chunks):
            self.log(f"Processing chunk {i+1}/{len(text_chunks)}")
            triples, new_relations = self._extract_from_chunk(
                chunk, entities, context, known_relations
            )
            all_triples.extend(triples)
            
            # Register new relations
            for rel in new_relations:
                self._register_relation(rel, document_id)
        
        # Propose triples for collaborative refinement
        if propose_to_agents and self.collab:
            all_triples = self._propose_and_refine(all_triples)
        
        # Store in memory
        if self.shared_memory:
            self._store_extractions(all_triples, document_id)
        
        self.log(f"Extracted {len(all_triples)} triples")
        return all_triples

    def _get_relevant_context(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get relevant context from shared memory."""
        if not self.shared_memory:
            return {}
        
        context = {
            "entity_histories": {},
            "related_triples": [],
            "document_context": [],
        }
        
        for entity in entities[:10]:  # Limit to avoid context overflow
            entity_context = self.shared_memory.get_entity_context(entity.id)
            if entity_context["memory_count"] > 0:
                context["entity_histories"][entity.id] = entity_context
        
        # Get recent semantic memories
        recent_memories = self.shared_memory.retrieve(
            memory_type=MemoryType.SEMANTIC,
            limit=20,
        )
        for mem in recent_memories:
            if "triples" in mem.content:
                context["related_triples"].extend(mem.content["triples"][:5])
        
        return context

    def _get_known_relations(self) -> List[Dict[str, Any]]:
        """Get previously discovered relation types."""
        relations = []
        for name, rel in self.discovered_relations.items():
            relations.append({
                "name": name,
                "definition": rel.definition,
                "examples": rel.examples[:3],
                "frequency": rel.frequency,
            })
        return sorted(relations, key=lambda r: r["frequency"], reverse=True)[:20]

    def _extract_from_chunk(
        self,
        text: str,
        entities: List[Entity],
        context: Dict[str, Any],
        known_relations: List[Dict[str, Any]],
    ) -> Tuple[List[Triple], List[DiscoveredRelation]]:
        """Extract relations from a single chunk."""
        
        # Build context-aware prompt
        entity_info = self._format_entities(entities)
        context_info = self._format_context(context)
        known_rel_info = self._format_known_relations(known_relations)
        
        system_msg = {
            "role": "system",
            "content": """You are an advanced relation extraction system operating in OPEN-WORLD mode.

Your task is to discover ALL meaningful relationships between entities, not just predefined types.

Guidelines:
1. Extract relationships that are explicitly stated or strongly implied
2. Create descriptive, semantic relation names (e.g., "founded_by", "is_ceo_of", "treats_condition")
3. Use lowercase with underscores for relation names
4. Provide clear definitions for any new relation types you introduce
5. Assign confidence scores based on how explicit the relationship is in the text
6. Consider temporal aspects if relevant (e.g., "was_ceo_of" vs "is_ceo_of")
7. Look for hierarchical, causal, temporal, spatial, and functional relationships

Be thorough but precise - only extract relationships with clear evidence.""",
        }
        
        user_prompt = f"""Extract all relationships from the following text.

ENTITIES:
{entity_info}

{context_info}

{known_rel_info}

TEXT:
{text}

Return a JSON object with:
{{
  "triples": [
    {{
      "subject": "entity name",
      "relation": "relation_name",
      "object": "entity name",
      "confidence": 0.0-1.0,
      "evidence": "quote or paraphrase from text",
      "temporal": "past/present/future/unknown"
    }}
  ],
  "new_relations": [
    {{
      "name": "relation_name",
      "definition": "clear definition of this relation type",
      "is_symmetric": false,
      "is_transitive": false
    }}
  ]
}}

Extract ALL meaningful relationships. Don't limit yourself to common relation types."""
        
        user_msg = {"role": "user", "content": user_prompt}
        
        try:
            response = chat_completion_json(
                messages=[system_msg, user_msg],
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=2500,
            )
            
            triples = []
            new_relations = []
            
            # Parse triples
            for item in response.get("triples", []):
                subject = item.get("subject", "").strip()
                relation = item.get("relation", "").strip()
                obj = item.get("object", "").strip()
                
                if not (subject and relation and obj):
                    continue
                
                # Validate entities exist
                entity_ids = {e.id.lower() for e in entities}
                entity_ids.update(e.id for e in entities)
                
                # Allow flexible matching
                subj_match = self._find_entity_match(subject, entities)
                obj_match = self._find_entity_match(obj, entities)
                
                if not subj_match or not obj_match:
                    self.log(f"Skipping: entities not found - {subject} or {obj}", level="WARNING")
                    continue
                
                triple = Triple(
                    subject=subj_match,
                    relation=relation.lower().replace(" ", "_"),
                    object=obj_match,
                    confidence=item.get("confidence", 0.7),
                    source=self.name,
                    metadata={
                        "evidence": item.get("evidence", ""),
                        "temporal": item.get("temporal", "unknown"),
                    },
                )
                triples.append(triple)
            
            # Parse new relations
            for item in response.get("new_relations", []):
                name = item.get("name", "").strip().lower().replace(" ", "_")
                if name and name not in self.discovered_relations:
                    new_relations.append(DiscoveredRelation(
                        name=name,
                        definition=item.get("definition", ""),
                    ))
            
            return triples, new_relations
            
        except Exception as e:
            self.log(f"Extraction failed: {e}", level="ERROR")
            return [], []

    def _find_entity_match(self, name: str, entities: List[Entity]) -> Optional[str]:
        """Find matching entity, allowing for fuzzy matching."""
        name_lower = name.lower().strip()
        
        # Exact match
        for e in entities:
            if e.id.lower() == name_lower or e.id == name:
                return e.id
        
        # Check aliases
        for e in entities:
            if any(label.lower() == name_lower for label in e.labels):
                return e.id
        
        # Partial match
        for e in entities:
            if name_lower in e.id.lower() or e.id.lower() in name_lower:
                return e.id
        
        return None

    def _register_relation(self, relation: DiscoveredRelation, document_id: Optional[str]) -> None:
        """Register a newly discovered relation type."""
        if relation.name in self.discovered_relations:
            existing = self.discovered_relations[relation.name]
            existing.frequency += 1
            if document_id and document_id not in existing.source_documents:
                existing.source_documents.append(document_id)
        else:
            if document_id:
                relation.source_documents = [document_id]
            self.discovered_relations[relation.name] = relation
            self.log(f"Discovered new relation type: {relation.name}")
        
        # Store in memory
        if self.shared_memory:
            self.shared_memory.store(
                memory_type=MemoryType.SEMANTIC,
                content={
                    "relation_type": relation.name,
                    "definition": relation.definition,
                    "frequency": self.discovered_relations[relation.name].frequency,
                },
                source=self.name,
            )

    def _propose_and_refine(self, triples: List[Triple]) -> List[Triple]:
        """Propose triples to other agents for refinement."""
        if not self.collab or not self.message_bus:
            return triples
        
        refined_triples = []
        
        for triple in triples:
            # Propose to other agents
            proposal_id = self.collab.propose_hypothesis(
                proposer=self.name,
                hypothesis={
                    "type": "triple",
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                },
                confidence=triple.confidence or 0.7,
                evidence=[triple.metadata.get("evidence", "")],
            )
            
            # For now, accept all (in real implementation, would wait for votes)
            refined_triples.append(triple)
        
        return refined_triples

    def _store_extractions(self, triples: List[Triple], document_id: Optional[str]) -> None:
        """Store extractions in shared memory."""
        if not self.shared_memory:
            return
        
        self.shared_memory.store(
            memory_type=MemoryType.SEMANTIC,
            content={
                "triples": [
                    {
                        "subject": t.subject,
                        "relation": t.relation,
                        "object": t.object,
                        "confidence": t.confidence,
                    }
                    for t in triples
                ],
                "document_id": document_id,
            },
            source=self.name,
        )

    def _format_entities(self, entities: List[Entity]) -> str:
        """Format entities for prompt."""
        lines = []
        for e in entities:
            type_str = f" ({e.type})" if e.type else ""
            aliases = f" [aliases: {', '.join(e.labels)}]" if e.labels else ""
            lines.append(f"- {e.id}{type_str}{aliases}")
        return "\n".join(lines)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        if not context or not context.get("entity_histories"):
            return ""
        
        lines = ["PRIOR CONTEXT:"]
        for entity_id, history in list(context["entity_histories"].items())[:5]:
            lines.append(f"- {entity_id}: seen in {history['memory_count']} previous contexts")
        
        if context.get("related_triples"):
            lines.append("\nPREVIOUS RELATED FACTS:")
            for t in context["related_triples"][:5]:
                lines.append(f"- ({t['subject']}) -[{t['relation']}]-> ({t['object']})")
        
        return "\n".join(lines)

    def _format_known_relations(self, relations: List[Dict[str, Any]]) -> str:
        """Format known relations for prompt."""
        if not relations:
            return ""
        
        lines = ["PREVIOUSLY DISCOVERED RELATION TYPES (you can use these or create new ones):"]
        for r in relations[:10]:
            lines.append(f"- {r['name']}: {r['definition']}")
        
        return "\n".join(lines)

    def get_relation_ontology(self) -> Dict[str, Any]:
        """Get the discovered relation ontology."""
        return {
            name: {
                "definition": rel.definition,
                "frequency": rel.frequency,
                "examples": rel.examples,
                "confidence": rel.confidence,
                "documents": rel.source_documents,
            }
            for name, rel in self.discovered_relations.items()
        }
