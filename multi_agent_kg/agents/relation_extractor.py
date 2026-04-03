"""
Relation Extractor Agent.

Implements consolidated triple extraction (reduced from 3 stages to 1 + gleaning):
1. Joint Triple Extraction: Extract complete (subject, relation, object) triples
2. Gleaning: Re-run to catch missed relationships (GraphRAG-style)

Features:
- Single-prompt extraction reduces error compounding (was 3 LLM calls, now 2)
- GraphRAG-style gleaning to recover missed relationships
- Constrained decoding via Pydantic schemas (eliminates JSON failures)
- Accepts prior triples from FastExtractor (GLiREL) for refinement
- Open-world relation discovery (not limited to predefined types)
- Self-consistency for confidence estimation
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
class DiscoveredRelationModel:
    """A relation type discovered during extraction."""
    name: str
    definition: str
    examples: List[Tuple[str, str, str]] = field(default_factory=list)
    frequency: int = 1
    confidence: float = 0.5
    source_documents: List[str] = field(default_factory=list)


JOINT_TRIPLE_EXTRACTION_PROMPT = """Think step by step: first identify relationships between entities, then output complete triples.

Given the text and list of entities below, extract ALL relationships as complete
(subject, relation, object) triples. For each triple, include a brief supporting
evidence snippet from the text.

DOMAIN: {domain}

ENTITIES FOUND:
{entities}

{suggested_types_section}

{prior_triples_section}

TEXT:
{text}

Instructions:
1. For each pair of entities that have a relationship, output the complete triple
2. Relation names should be descriptive UPPER_SNAKE_CASE (e.g., CAUSES, TREATED_BY)
3. Include both explicit relationships stated in the text AND implicit ones that can be reasonably inferred
4. Each triple must reference entities from the provided list

Return a JSON object with a "triples" list."""


RELATION_GLEANING_PROMPT = """The following triples were already extracted from this text.
Review the text carefully and find any relationships that were MISSED.

Focus on:
- Implicit or indirect relationships
- Causal relationships not explicitly stated
- Hierarchical relationships (part-of, is-a)
- Temporal relationships (before, after, during)
- Relationships involving entities that appear in few or no triples

ALREADY FOUND TRIPLES:
{found_triples}

ENTITIES:
{entities}

TEXT:
{text}

Return ONLY NEW triples that were missed. Do not repeat already-found triples.
Return a JSON object with a "triples" list. If no new triples are found, return {{"triples": []}}."""


class RelationExtractor(BaseAgent):
    """
    Relation Extractor Agent - Consolidated triple extraction with gleaning.

    Pipeline (reduced from 3 stages to 1 + gleaning):
    1. Joint Triple Extraction: Complete (subject, relation, object) in one pass
    2. Gleaning: Re-run to catch missed relationships

    Uses SharedMemory to:
    - Track discovered relation types across documents
    - Store extracted triples for cross-reference
    - Post novel relations to blackboard for voting

    Uses MessageBus to:
    - Receive domain info and entities
    - Send triples to downstream agents
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
        max_gleanings: int = 1,
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
        self.max_gleanings = max_gleanings

        # Track discovered relation types
        self.discovered_relations: Dict[str, DiscoveredRelationModelModel] = {}

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
        prior_triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract relations using consolidated single-pass + gleaning.

        Args:
            context: Processing context
            segments: Document segments
            entities: Extracted entities
            domain_config: Domain configuration
            prior_triples: Baseline triples from FastExtractor (GLiREL)

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

        # Batch segments to reduce LLM calls (5 segments per batch)
        BATCH_SIZE = 5
        all_triples: List[Dict[str, Any]] = []
        low_confidence_triples: List[Dict[str, Any]] = []

        raw_segments = []
        if segments:
            raw_segments = [(s.get("text", ""), s.get("segment_id", "")) for s in segments]
        elif context.text:
            raw_segments = [(context.text, f"{context.document_id}_full")]

        # Group segments into batches
        batches = []
        for i in range(0, len(raw_segments), BATCH_SIZE):
            batches.append(raw_segments[i:i + BATCH_SIZE])

        print(f"  Processing {len(raw_segments)} segments in {len(batches)} batches")

        for batch_idx, batch in enumerate(batches):
            combined_text = "\n\n---\n\n".join(text for text, _ in batch if text)
            batch_ids = [sid for _, sid in batch]

            if not combined_text or len(combined_text) < 20:
                continue

            # Per-batch prior triples
            batch_priors = []
            if prior_triples:
                batch_priors = [
                    t for t in prior_triples
                    if t.get("source_segment") in batch_ids
                    or not t.get("source_segment")
                ][:20]  # Cap to avoid prompt overflow

            # Stage 1: Joint triple extraction (one call per batch)
            triples = self._stage1_joint_extraction(
                combined_text, entities, suggested_types, context.domain, batch_priors,
            )

            # Stage 2: Gleaning — one pass per batch
            for _ in range(self.max_gleanings):
                gleaned = self._stage2_gleaning(combined_text, entities, triples)
                if not gleaned:
                    break
                triples.extend(gleaned)

            # Track discovered relation types
            for triple in triples:
                rel_type = triple.get("relation", "")
                if rel_type and rel_type not in self.discovered_relations:
                    self._register_new_relation(
                        {"relation_type": rel_type, "definition": ""},
                        context.document_id,
                    )

            # Add batch info and separate by confidence
            batch_label = batch_ids[0] if batch_ids else f"batch_{batch_idx}"
            for triple in triples:
                if not triple.get("source_segment"):
                    triple["source_segment"] = batch_label
                triple["document_id"] = context.document_id

                if triple.get("confidence", 0) >= self.quality_threshold:
                    all_triples.append(triple)
                else:
                    low_confidence_triples.append(triple)

            print(f"    Batch {batch_idx + 1}/{len(batches)}: {len(triples)} triples")

        # Handle low confidence triples
        print(f"\n[RELATION EXTRACTOR DEBUG]")
        print(f"  Total extracted: {len(all_triples) + len(low_confidence_triples)}")
        print(f"  High confidence (>={self.quality_threshold}): {len(all_triples)}")
        print(f"  Low confidence (<{self.quality_threshold}): {len(low_confidence_triples)}")
        print(f"  Relation types discovered: {len(self.discovered_relations)}")

        if low_confidence_triples:
            self._handle_low_confidence_triples(low_confidence_triples, context)

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
            f"{len(self.discovered_relations)} relation types tracked"
        )

        return ExtractionResult(
            items=all_triples,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_triples),
                "new_relations_discovered": len(self.discovered_relations),
                "relation_types_used": list(set(t.get("relation", "") for t in all_triples)),
                "prior_triples_used": len(prior_triples) if prior_triples else 0,
            },
            needs_escalation=len(low_confidence_triples) > 0,
            escalation_reason=self._get_escalation_reason(low_confidence_triples, []),
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

    def _stage1_joint_extraction(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        suggested_types: List[str],
        domain: Optional[str],
        prior_triples: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Stage 1: Joint triple extraction in a single prompt."""
        from multi_agent_kg.schemas.extraction_schemas import TripleExtractionResponse

        entities_str = ", ".join(
            f'"{e.get("text", str(e))}" ({e.get("type", "?")})'
            for e in entities[:50]
        )

        suggested_section = ""
        if suggested_types:
            suggested_section = (
                f"Suggested relation types: {', '.join(suggested_types)}\n"
                f"You may also discover new relation types based on the content."
            )

        prior_section = ""
        if prior_triples:
            prior_list = "\n".join(
                f'  ({t.get("subject", "?")}) --[{t.get("relation", "?")}]--> ({t.get("object", "?")})'
                for t in prior_triples[:20]
            )
            prior_section = (
                f"A preliminary scan found these relationships:\n{prior_list}\n"
                f"Review them, correct errors, and add missed relationships."
            )

        prompt = JOINT_TRIPLE_EXTRACTION_PROMPT.format(
            text=text,
            entities=entities_str,
            suggested_types_section=suggested_section,
            prior_triples_section=prior_section,
            domain=domain or "general",
        )

        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at extracting relationships between entities. "
                    "Output complete triples with evidence."
                ),
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
            triples = self._normalize_triple_list(result.get("triples", []))
            for t in triples:
                t["confidence"] = (t.get("confidence", 0.7) + confidence) / 2
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at extracting relationships between entities. "
                    "Output complete triples with evidence."
                ),
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
                response_schema=TripleExtractionResponse,
            )
            triples = self._normalize_triple_list(result.get("triples", []))

        return triples

    def _stage2_gleaning(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        found_triples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Gleaning -- find missed relationships."""
        from multi_agent_kg.schemas.extraction_schemas import TripleExtractionResponse

        if not found_triples:
            return []

        found_list = "\n".join(
            f'  ({t.get("subject", "?")}) --[{t.get("relation", "?")}]--> ({t.get("object", "?")})'
            for t in found_triples[:30]
        )
        entities_str = ", ".join(
            f'"{e.get("text", str(e)) if isinstance(e, dict) else str(e)}"'
            for e in entities[:40]
        )

        prompt = RELATION_GLEANING_PROMPT.format(
            text=text,
            entities=entities_str,
            found_triples=found_list,
        )

        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at finding missed relationships. Only return NEW triples.",
            tier=ModelTier.MEDIUM,
            max_tokens=2048,
            response_schema=TripleExtractionResponse,
        )

        new_triples = self._normalize_triple_list(result.get("triples", []))
        for t in new_triples:
            t["confidence"] = min(t.get("confidence", 0.6), 0.7)
            t["gleaned"] = True
        return new_triples

    @staticmethod
    def _normalize_triple_list(raw: list) -> List[Dict[str, Any]]:
        """Normalise triples that may have non-standard keys or be strings."""
        normalised = []
        for item in raw:
            if isinstance(item, str):
                # Skip bare strings — can't recover a triple from just a string
                continue
            if not isinstance(item, dict):
                continue
            # Accept common aliases
            if "subject" not in item:
                item["subject"] = item.pop("head", item.pop("head_entity", item.pop("source", "")))
            if "object" not in item:
                item["object"] = item.pop("tail", item.pop("tail_entity", item.pop("target", "")))
            if "relation" not in item:
                item["relation"] = item.pop("predicate", item.pop("relation_type", item.pop("type", "")))
            # Only keep triples with all three parts
            if item.get("subject") and item.get("relation") and item.get("object"):
                normalised.append(item)
        return normalised

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
            self.discovered_relations[name] = DiscoveredRelationModel(
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
        
        # Basic validation — no hardcoded relation patterns; the system discovers all relations
        if not subject or not relation or not obj:
            return VoteType.REJECT, 0.9, "Triple missing subject, relation, or object"

        # Check if relation type was previously discovered (learned, not hardcoded)
        known_relations = list(self.discovered_relations.keys())
        if relation.lower() in [r.lower() for r in known_relations]:
            return VoteType.ACCEPT, 0.8, f"Previously discovered relation type: {relation}"

        # New relation type — accept it (the system learns new relations)
        return VoteType.WEAK_ACCEPT, 0.65, f"New relation type discovered: {relation}"

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

    def get_discovered_relations(self) -> Dict[str, DiscoveredRelationModel]:
        """Get all discovered relation types."""
        return self.discovered_relations
