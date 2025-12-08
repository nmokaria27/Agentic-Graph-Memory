"""
Evidence Linker Agent.

Responsible for:
- Linking extracted triples to source evidence
- Computing evidence quality scores
- Cross-referencing with prior extractions
- Building provenance chains

This is the last worker agent before coordinators validate.
"""

from typing import Any, Dict, List, Optional
import json

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig


EVIDENCE_LINKING_PROMPT = """Link each triple to its supporting evidence in the text.

TEXT:
{text}

TRIPLES TO LINK:
{triples_json}

For each triple, find:
1. The exact sentence(s) that support it
2. The strength of the evidence (explicit, implicit, or inferred)
3. Any contradicting evidence

Return:
{{
    "linked_triples": [
        {{
            "triple": {{
                "subject": "<subject>",
                "relation": "<relation>",
                "object": "<object>"
            }},
            "evidence_sentences": ["<sentence1>", "<sentence2>", ...],
            "evidence_type": "<explicit|implicit|inferred>",
            "evidence_strength": <0.0-1.0>,
            "contradictions": ["<any contradicting text>"],
            "char_positions": [
                {{"start": <start>, "end": <end>}}
            ]
        }}
    ]
}}"""


CROSS_REFERENCE_PROMPT = """Check if these triples are supported by prior knowledge.

NEW TRIPLES:
{new_triples}

PRIOR KNOWLEDGE (from previous documents):
{prior_knowledge}

For each new triple, determine:
1. Does prior knowledge support, contradict, or add to this?
2. If supported, by which prior facts?
3. If contradicted, what is the conflict?

Return:
{{
    "cross_references": [
        {{
            "triple": {{
                "subject": "<subject>",
                "relation": "<relation>",
                "object": "<object>"
            }},
            "status": "<supported|contradicted|novel|refined>",
            "prior_evidence": ["<related prior fact>", ...],
            "confidence_adjustment": <-0.3 to +0.3>,
            "notes": "<explanation>"
        }}
    ]
}}"""


class EvidenceLinker(BaseAgent):
    """
    Evidence Linker Agent - Links triples to source evidence.
    
    Responsibilities:
    1. Find supporting sentences for each triple
    2. Classify evidence type (explicit, implicit, inferred)
    3. Cross-reference with prior knowledge
    4. Adjust confidence based on evidence quality
    
    Uses SharedMemory to:
    - Retrieve prior extractions for cross-reference
    - Store evidence links for provenance
    
    Uses MessageBus to:
    - Receive triples from RelationExtractor
    - Send linked triples to validators
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        enable_cross_reference: bool = True,
    ):
        super().__init__(
            name="EvidenceLinker",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.enable_cross_reference = enable_cross_reference

    def run(
        self,
        context: AgentContext,
        triples: Optional[List[Dict[str, Any]]] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Link triples to supporting evidence.
        
        Args:
            context: Processing context
            triples: Extracted triples to link
            segments: Document segments for evidence search
            
        Returns:
            ExtractionResult with evidence-linked triples
        """
        self.stats["calls"] += 1
        
        triples = triples or context.relations or []
        
        if not triples:
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={"error": "No triples to link"},
            )
        
        # Get full text for evidence search
        if segments:
            full_text = " ".join(s.get("text", "") for s in segments)
        else:
            full_text = context.text
        
        # Stage 1: Link to source evidence
        linked_triples = self._link_to_evidence(triples, full_text)
        
        # Stage 2: Cross-reference with prior knowledge
        if self.enable_cross_reference:
            prior_knowledge = self._get_prior_knowledge()
            if prior_knowledge:
                linked_triples = self._cross_reference(linked_triples, prior_knowledge)
        
        # Calculate final confidence for each triple
        for triple in linked_triples:
            triple["final_confidence"] = self._calculate_final_confidence(triple)
        
        # Separate by quality
        high_quality = []
        needs_review = []
        
        for triple in linked_triples:
            if triple.get("final_confidence", 0) >= self.quality_threshold:
                high_quality.append(triple)
            else:
                needs_review.append(triple)
        
        # Handle low quality triples
        if needs_review:
            self._handle_needs_review(needs_review, context)
        
        # Store evidence links
        if self.shared_memory:
            self._store_evidence_links(linked_triples, context.document_id)
        
        # Calculate overall confidence
        if linked_triples:
            avg_confidence = sum(t.get("final_confidence", 0.5) for t in linked_triples) / len(linked_triples)
        else:
            avg_confidence = 0.0
        
        self.log(
            f"Linked {len(high_quality)} high-quality triples, "
            f"{len(needs_review)} need review"
        )
        
        return ExtractionResult(
            items=linked_triples,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "high_quality_count": len(high_quality),
                "needs_review_count": len(needs_review),
                "cross_referenced": self.enable_cross_reference,
            },
            needs_escalation=len(needs_review) > 0,
            escalation_reason=f"{len(needs_review)} triples need review" if needs_review else None,
        )

    def _link_to_evidence(
        self,
        triples: List[Dict[str, Any]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Link triples to supporting evidence in text."""
        if not triples or not text:
            return triples
        
        # Process in batches to avoid token limits
        batch_size = 10
        all_linked = []
        
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            
            triples_json = json.dumps([
                {
                    "subject": t.get("subject", ""),
                    "relation": t.get("relation", ""),
                    "object": t.get("object", ""),
                }
                for t in batch
            ], indent=2)
            
            prompt = EVIDENCE_LINKING_PROMPT.format(
                text=text[:4000],  # Limit text length
                triples_json=triples_json,
            )
            
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at finding evidence for claims. Be precise about source sentences.",
                tier=ModelTier.MEDIUM,
            )
            
            linked = result.get("linked_triples", [])
            
            # Merge back with original triple data
            for j, linked_triple in enumerate(linked):
                if j < len(batch):
                    merged = {**batch[j], **linked_triple}
                    all_linked.append(merged)
        
        return all_linked

    def _get_prior_knowledge(self) -> List[Dict[str, Any]]:
        """Get prior knowledge from memory and knowledge graph."""
        prior = []
        
        # From knowledge graph
        if self.knowledge_graph:
            for triple_id, triple in list(self.knowledge_graph.triples.items())[:50]:
                prior.append({
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                    "confidence": triple.confidence,
                })
        
        # From shared memory
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=20)
            for mem in memories:
                if "triples" in mem.content:
                    prior.extend(mem.content["triples"][:10])
        
        return prior

    def _cross_reference(
        self,
        triples: List[Dict[str, Any]],
        prior_knowledge: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cross-reference triples with prior knowledge."""
        if not triples or not prior_knowledge:
            return triples
        
        new_triples_json = json.dumps([
            {
                "subject": t.get("subject", t.get("triple", {}).get("subject", "")),
                "relation": t.get("relation", t.get("triple", {}).get("relation", "")),
                "object": t.get("object", t.get("triple", {}).get("object", "")),
            }
            for t in triples[:20]
        ], indent=2)
        
        prior_json = json.dumps(prior_knowledge[:30], indent=2)
        
        prompt = CROSS_REFERENCE_PROMPT.format(
            new_triples=new_triples_json,
            prior_knowledge=prior_json,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at knowledge integration. Check for consistency with prior facts.",
            tier=ModelTier.MEDIUM,
        )
        
        cross_refs = result.get("cross_references", [])
        
        # Apply cross-reference results
        for i, xref in enumerate(cross_refs):
            if i < len(triples):
                triples[i]["cross_reference_status"] = xref.get("status", "novel")
                triples[i]["prior_evidence"] = xref.get("prior_evidence", [])
                triples[i]["confidence_adjustment"] = xref.get("confidence_adjustment", 0)
                triples[i]["cross_reference_notes"] = xref.get("notes", "")
        
        return triples

    def _calculate_final_confidence(
        self,
        triple: Dict[str, Any],
    ) -> float:
        """Calculate final confidence based on all factors."""
        # Start with extraction confidence
        base_confidence = triple.get("confidence", 0.7)
        
        # Adjust based on evidence type
        evidence_type = triple.get("evidence_type", "inferred")
        evidence_multiplier = {
            "explicit": 1.0,
            "implicit": 0.85,
            "inferred": 0.7,
        }.get(evidence_type, 0.7)
        
        # Adjust based on evidence strength
        evidence_strength = triple.get("evidence_strength", 0.7)
        
        # Adjust based on cross-reference
        xref_status = triple.get("cross_reference_status", "novel")
        xref_adjustment = {
            "supported": 0.15,
            "contradicted": -0.3,
            "novel": 0.0,
            "refined": 0.1,
        }.get(xref_status, 0)
        
        # Check for contradictions
        contradictions = triple.get("contradictions", [])
        if contradictions:
            xref_adjustment -= 0.1 * len(contradictions)
        
        # Calculate final
        final = (
            base_confidence * evidence_multiplier * evidence_strength 
            + xref_adjustment
        )
        
        return max(0.0, min(1.0, final))

    def _handle_needs_review(
        self,
        triples: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle triples that need review."""
        # Post to blackboard
        for triple in triples[:10]:
            self.post_hypothesis(
                hypothesis={
                    "subject": triple.get("subject", triple.get("triple", {}).get("subject", "")),
                    "relation": triple.get("relation", triple.get("triple", {}).get("relation", "")),
                    "object": triple.get("object", triple.get("triple", {}).get("object", "")),
                },
                confidence=triple.get("final_confidence", 0.5),
                evidence=triple.get("evidence_sentences", []),
            )
        
        # Escalate
        self.escalate_to_coordinator(
            reason="Triples with weak evidence",
            items=triples,
            context={
                "document_id": context.document_id,
            },
        )

    def _store_evidence_links(
        self,
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store evidence links in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "evidence_linked_triples": triples,
                "document_id": document_id,
            },
        )
