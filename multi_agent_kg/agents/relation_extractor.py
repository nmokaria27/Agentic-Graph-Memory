"""
Relation Extractor Agent.

Implements RHF (Relation-Head-First) multi-stage extraction:
1. Relation Identification: Find relation types present in text
2. Head Entity Binding: Bind relations to head (subject) entities  
3. Tail Entity Binding: Complete triples with tail (object) entities

Features:
- Open-world relation discovery (not limited to predefined types)
- Self-consistency for confidence estimation
- Relation type learning via SharedMemory
- Blackboard voting for novel relations
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import json

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType


@dataclass
class DiscoveredRelation:
    """A relation type discovered during extraction."""
    name: str
    definition: str
    examples: List[Tuple[str, str, str]] = field(default_factory=list)
    frequency: int = 1
    confidence: float = 0.5
    source_documents: List[str] = field(default_factory=list)


RELATION_IDENTIFICATION_PROMPT = """Identify all relation types present in the following text.

DOMAIN: {domain}
SUGGESTED RELATION TYPES: {suggested_types}

TEXT:
{text}

ENTITIES FOUND:
{entities}

Instructions:
1. Identify explicit and implicit relations between entities
2. Use suggested relation types when appropriate
3. Propose NEW relation types if needed (open-world extraction)
4. For new types, provide a clear definition

Return:
{{
    "relations_found": [
        {{
            "relation_type": "<relation name>",
            "is_new_type": <true/false>,
            "definition": "<definition if new type>",
            "count_in_text": <approximate count>
        }}
    ]
}}"""


HEAD_BINDING_PROMPT = """For each relation type, identify the HEAD (subject) entities.

TEXT:
{text}

ENTITIES:
{entities}

RELATION TYPES TO BIND:
{relation_types}

For each relation occurrence, identify what entity is the SUBJECT (head) of that relation.

Return:
{{
    "head_bindings": [
        {{
            "relation_type": "<relation>",
            "head_entity": "<subject entity text>",
            "head_entity_id": "<entity id if available>",
            "context": "<sentence or phrase containing this>",
            "confidence": <0.0-1.0>
        }}
    ]
}}"""


TAIL_BINDING_PROMPT = """Complete the triples by adding TAIL (object) entities.

TEXT:
{text}

ENTITIES:
{entities}

HEAD BINDINGS (subject-relation pairs):
{head_bindings}

For each head binding, identify what entity is the OBJECT (tail) of that relation.

Return:
{{
    "triples": [
        {{
            "subject": "<head entity>",
            "subject_id": "<head entity id>",
            "relation": "<relation type>",
            "object": "<tail entity>",
            "object_id": "<tail entity id>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text snippet>"
        }}
    ]
}}"""


class RelationExtractor(BaseAgent):
    """
    Relation Extractor Agent - RHF multi-stage relation extraction.
    
    Pipeline (Relation-Head-First):
    1. Relation Identification: Find what relations exist in text
    2. Head Entity Binding: Bind relations to subject entities
    3. Tail Entity Binding: Complete triples with object entities
    
    Uses SharedMemory to:
    - Track discovered relation types across documents
    - Store extracted triples for cross-reference
    - Post novel relations to blackboard for voting
    
    Uses MessageBus to:
    - Receive domain info and entities
    - Send triples to EvidenceLinker
    - Escalate low-confidence extractions
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        use_self_consistency: bool = True,
        n_consistency_samples: int = 3,
        enable_open_world: bool = True,
    ):
        super().__init__(
            name="RelationExtractor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.use_self_consistency = use_self_consistency
        self.n_consistency_samples = n_consistency_samples
        self.enable_open_world = enable_open_world
        
        # Track discovered relation types
        self.discovered_relations: Dict[str, DiscoveredRelation] = {}

    def _normalize_relation_types(self, relation_types_raw: Any) -> List[str]:
        """Normalize relation types from various formats to List[str]."""
        if not relation_types_raw:
            return []
        
        if not isinstance(relation_types_raw, list):
            return []
        
        normalized = []
        for rt in relation_types_raw:
            if isinstance(rt, dict):
                # Extract 'type' field from dict format
                if "type" in rt:
                    normalized.append(rt["type"])
            elif isinstance(rt, str):
                normalized.append(rt)
        
        return normalized

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract relations using RHF pipeline.
        
        Args:
            context: Processing context
            segments: Document segments
            entities: Extracted entities
            domain_config: Domain configuration
            
        Returns:
            ExtractionResult with extracted triples
        """
        self.stats["calls"] += 1
        
        # Use entities from context if not provided
        entities = entities or context.entities or []
        
        # Get relation types from domain or discovered
        suggested_types = self._get_suggested_relation_types(domain_config)
        
        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "relation_types" in msg.content:
                    suggested_types = self._normalize_relation_types(msg.content["relation_types"])
        
        # Process segments or full text
        all_triples = []
        low_confidence_triples = []
        new_relations_discovered = []
        
        texts_to_process = []
        if segments:
            texts_to_process = [(s.get("text", ""), s.get("segment_id")) for s in segments]
        elif context.text:
            texts_to_process = [(context.text, f"{context.document_id}_full")]
        
        for text, segment_id in texts_to_process:
            if not text or len(text) < 20:
                continue
            
            # RHF Pipeline
            # Stage 1: Relation Identification
            relations_found = self._stage1_identify_relations(
                text, 
                entities,
                suggested_types,
                context.domain,
            )
            
            # Track new relation types
            for rel in relations_found:
                if rel.get("is_new_type"):
                    new_relations_discovered.append(rel)
                    self._register_new_relation(rel, context.document_id)
            
            relation_types = [r["relation_type"] for r in relations_found]
            
            if not relation_types:
                continue
            
            # Stage 2: Head Entity Binding
            head_bindings = self._stage2_head_binding(
                text,
                entities,
                relation_types,
            )
            
            if not head_bindings:
                continue
            
            # Stage 3: Tail Entity Binding
            triples = self._stage3_tail_binding(
                text,
                entities,
                head_bindings,
            )
            
            # Add segment info and separate by confidence
            for triple in triples:
                triple["source_segment"] = segment_id
                triple["document_id"] = context.document_id
                
                if triple.get("confidence", 0) >= self.quality_threshold:
                    all_triples.append(triple)
                else:
                    low_confidence_triples.append(triple)
        
        # Handle low confidence triples
        if low_confidence_triples:
            self._handle_low_confidence_triples(
                low_confidence_triples,
                context,
            )
        
        # Handle new relation types
        if new_relations_discovered:
            self._handle_new_relations(
                new_relations_discovered,
                context,
            )
        
        # Store results
        if self.shared_memory:
            self._store_triples(all_triples, context.document_id)
        
        # Calculate overall confidence
        if all_triples:
            avg_confidence = sum(t.get("confidence", 0.5) for t in all_triples) / len(all_triples)
        else:
            avg_confidence = 0.0
        
        self.log(
            f"Extracted {len(all_triples)} triples, "
            f"{len(new_relations_discovered)} new relation types discovered"
        )
        
        return ExtractionResult(
            items=all_triples,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_triples),
                "new_relations_discovered": len(new_relations_discovered),
                "relation_types_used": list(set(t.get("relation", "") for t in all_triples)),
            },
            needs_escalation=len(low_confidence_triples) > 0 or len(new_relations_discovered) > 0,
            escalation_reason=self._get_escalation_reason(low_confidence_triples, new_relations_discovered),
        )

    def _get_suggested_relation_types(
        self,
        domain_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Get suggested relation types from domain and discovered."""
        types = []
        
        # From domain config - normalize from dict format
        if domain_config:
            relation_types_raw = domain_config.get("relation_types", [])
            types.extend(self._normalize_relation_types(relation_types_raw))
        
        # From discovered relations (high frequency)
        for name, rel in self.discovered_relations.items():
            if rel.frequency >= 2 and rel.confidence >= 0.6:
                types.append(name)
        
        # From shared memory
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            for mem in memories:
                if "discovered_relations" in mem.content:
                    types.extend(mem.content["discovered_relations"])
        
        return list(set(types))

    def _stage1_identify_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        suggested_types: List[str],
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Stage 1: Identify relation types in text."""
        entities_str = ", ".join(e.get("text", str(e)) for e in entities[:20])
        
        prompt = RELATION_IDENTIFICATION_PROMPT.format(
            text=text,
            entities=entities_str,
            suggested_types=", ".join(suggested_types) if suggested_types else "none provided (discover new types)",
            domain=domain or "general",
        )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert at identifying relations between entities. Be thorough but precise.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at identifying relations between entities. Be thorough but precise.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
        
        return result.get("relations_found", [])

    def _stage2_head_binding(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Bind relations to head (subject) entities."""
        if not relation_types:
            return []
        
        entities_json = json.dumps(entities[:30], indent=2)
        
        prompt = HEAD_BINDING_PROMPT.format(
            text=text,
            entities=entities_json,
            relation_types=", ".join(relation_types),
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at identifying subject-relation pairs in text.",
            tier=ModelTier.MEDIUM,
            max_tokens=4096,
        )
        
        return result.get("head_bindings", [])

    def _stage3_tail_binding(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        head_bindings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 3: Complete triples with tail (object) entities."""
        if not head_bindings:
            return []
        
        entities_json = json.dumps(entities[:30], indent=2)
        head_bindings_json = json.dumps(head_bindings, indent=2)
        
        prompt = TAIL_BINDING_PROMPT.format(
            text=text,
            entities=entities_json,
            head_bindings=head_bindings_json,
        )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert at completing relation triples. Be precise about object entities.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
            
            # Adjust confidences based on consistency
            triples = result.get("triples", [])
            for t in triples:
                # Combine LLM confidence with self-consistency
                t["confidence"] = (t.get("confidence", 0.7) + confidence) / 2
            return triples
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at completing relation triples. Be precise about object entities.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            return result.get("triples", [])

    def _register_new_relation(
        self,
        relation: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Register a newly discovered relation type."""
        name = relation.get("relation_type", "")
        if not name:
            return
        
        if name in self.discovered_relations:
            self.discovered_relations[name].frequency += 1
            self.discovered_relations[name].source_documents.append(document_id)
        else:
            self.discovered_relations[name] = DiscoveredRelation(
                name=name,
                definition=relation.get("definition", ""),
                frequency=1,
                confidence=0.5,
                source_documents=[document_id],
            )

    def _handle_low_confidence_triples(
        self,
        triples: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle low confidence triples via escalation."""
        # Post to blackboard for voting
        for triple in triples[:10]:
            self.post_hypothesis(
                hypothesis={
                    "subject": triple.get("subject"),
                    "relation": triple.get("relation"),
                    "object": triple.get("object"),
                },
                confidence=triple.get("confidence", 0.5),
                evidence=[triple.get("evidence", "")],
            )
        
        # Submit to deliberation for multi-agent voting
        for triple in triples[:10]:  # Limit
            self.submit_for_deliberation(
                hypothesis_type="triple",
                content=triple,
                confidence=triple.get("confidence", 0.5),
                evidence=[triple.get("evidence", "")],
                document_id=context.document_id,
            )
        
        # Escalate to coordinator
        self.escalate_to_coordinator(
            reason="Low confidence relation extractions submitted for deliberation",
            items=triples,
            context={
                "document_id": context.document_id,
                "domain": context.domain,
            },
        )

    def _handle_new_relations(
        self,
        new_relations: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle newly discovered relation types via deliberation."""
        # Submit new relation types for community voting
        for rel in new_relations:
            self.submit_for_deliberation(
                hypothesis_type="relation_type",
                content={
                    "relation_type": rel.get("relation_type"),
                    "definition": rel.get("definition"),
                },
                confidence=0.6,
                evidence=[context.document_id],
                document_id=context.document_id,
            )
        
        # Store in memory for future reference
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.SEMANTIC,
                content={
                    "discovered_relations": [r.get("relation_type") for r in new_relations],
                    "definitions": {r.get("relation_type"): r.get("definition") for r in new_relations},
                },
            )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        RelationExtractor's logic for voting on hypotheses.
        
        Can vote on:
        - entity: Abstain (not our specialty)
        - relation: Check if relation type is valid
        - triple: Check if relation makes semantic sense
        - relation_type: Evaluate new relation type proposals
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        if hypothesis_type == "entity":
            # Entities are not our specialty
            return VoteType.ABSTAIN, 0.5, "RelationExtractor focuses on relations"
        elif hypothesis_type == "relation" or hypothesis_type == "triple":
            return self._vote_on_triple(hypothesis_content, context)
        elif hypothesis_type == "relation_type":
            return self._vote_on_relation_type(hypothesis_content, context)
        
        return VoteType.ABSTAIN, 0.5, "RelationExtractor cannot evaluate this hypothesis type"

    def _vote_on_triple(
        self,
        triple: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on a triple hypothesis."""
        from multi_agent_kg.core.deliberation import VoteType
        
        subject = triple.get("subject", "")
        relation = triple.get("relation", "") or triple.get("relation_type", "")
        obj = triple.get("object", "")
        
        # Basic validation
        if not subject or not relation or not obj:
            return VoteType.REJECT, 0.9, "Triple missing subject, relation, or object"
        
        # Check if relation type is known
        known_relations = list(self.discovered_relations.keys()) + self.domain_relations.get("general", [])
        if relation.lower() in [r.lower() for r in known_relations]:
            return VoteType.ACCEPT, 0.8, f"Known relation type: {relation}"
        
        # Check for common sense relation patterns
        relation_lower = relation.lower().replace("_", " ")
        common_patterns = ["is a", "works for", "located in", "part of", "born in", 
                          "founded", "married to", "has", "owns", "created", "leads"]
        if any(p in relation_lower for p in common_patterns):
            return VoteType.WEAK_ACCEPT, 0.7, f"Relation follows common pattern"
        
        # Unknown relation - weak reject
        return VoteType.WEAK_REJECT, 0.6, f"Unknown relation type: {relation}"

    def _vote_on_relation_type(
        self,
        relation_type: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on a new relation type proposal."""
        from multi_agent_kg.core.deliberation import VoteType
        
        rel_name = relation_type.get("relation_type", "")
        definition = relation_type.get("definition", "")
        
        if not rel_name:
            return VoteType.REJECT, 0.9, "No relation type name provided"
        
        if not definition:
            return VoteType.WEAK_REJECT, 0.7, "New relation type needs a definition"
        
        # Check if relation already exists
        if rel_name in self.discovered_relations:
            return VoteType.REJECT, 0.8, f"Relation type '{rel_name}' already exists"
        
        # Accept if well-defined
        if len(definition) > 20:
            return VoteType.WEAK_ACCEPT, 0.7, "New relation type with good definition"
        
        return VoteType.WEAK_REJECT, 0.6, "Definition too short for new relation type"

    def _store_triples(
        self,
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store extracted triples in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "triples": triples,
                "document_id": document_id,
            },
        )

    def _get_escalation_reason(
        self,
        low_confidence: List[Dict[str, Any]],
        new_relations: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Generate escalation reason."""
        reasons = []
        if low_confidence:
            reasons.append(f"{len(low_confidence)} low confidence triples")
        if new_relations:
            reasons.append(f"{len(new_relations)} new relation types")
        return ", ".join(reasons) if reasons else None

    def get_discovered_relations(self) -> Dict[str, DiscoveredRelation]:
        """Get all discovered relation types."""
        return self.discovered_relations
