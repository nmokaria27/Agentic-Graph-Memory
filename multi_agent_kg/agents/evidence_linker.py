"""
Evidence Linker Agent.

Responsible for:
- Linking extracted triples to source evidence
- Computing evidence quality scores
- Cross-referencing with prior extractions
- Building provenance chains

This is the last worker agent before coordinators validate.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import re

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

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType


EVIDENCE_LINKING_PROMPT = """Link each triple to its supporting evidence in the text.

DOMAIN: {domain}

TEXT:
{text}

TRIPLES TO LINK:
{triples_json}

For each triple, find:
1. The exact sentence(s) that support it
2. The strength of the evidence (explicit, implicit, or inferred)
3. Any contradicting evidence

For domain-specific triples, domain knowledge counts as implicit evidence.

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
        strict_source_only: bool = False,
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
        self.strict_source_only = strict_source_only

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

        # Extract domain_config if provided
        domain_config = kwargs.get("domain_config")
        self._current_domain = domain_config.get("domain", "general") if domain_config else (context.domain or "general")

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
        
        # Stage 2: Cross-reference with prior knowledge. Strict source-only
        # ablations intentionally disable this so "source-supported" means
        # supported by the current document, not by prior KG memory.
        if self.enable_cross_reference and not self.strict_source_only:
            prior_knowledge = self._get_prior_knowledge()
            if prior_knowledge:
                linked_triples = self._cross_reference(linked_triples, prior_knowledge)

        if self.strict_source_only:
            linked_triples = self._enforce_strict_source_support(linked_triples, full_text)
        
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
        # Calculate batch size based on model capacity and item size
        avg_triple_size = 100  # Approximate tokens per triple with evidence
        prompt_overhead = 500  # Base prompt tokens
        model_input_limit = 16000  # Conservative context window estimate
        model_output_limit = 4096  # max_tokens for response
        
        # Calculate how many triples can fit while leaving room for output
        available_for_data = model_input_limit - prompt_overhead - len(text) // 4
        batch_size = max(3, min(20, available_for_data // avg_triple_size))
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
                domain=getattr(self, '_current_domain', 'general'),
            )
            
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at finding evidence for claims. Be precise about source sentences.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            
            linked = (
                result
                if isinstance(result, list)
                else result.get("linked_triples", []) if isinstance(result, dict)
                else []
            )
            linked_by_key = {}
            for item in linked:
                if not isinstance(item, dict):
                    continue
                triple_payload = item.get("triple", item) if isinstance(item, dict) else {}
                key = (
                    str(triple_payload.get("subject", "")).strip().lower(),
                    str(triple_payload.get("relation", "")).strip().lower(),
                    str(triple_payload.get("object", "")).strip().lower(),
                )
                linked_by_key[key] = item
            
            # Merge back with original triple data. Never drop a candidate just
            # because the LLM omitted it; missing links become unsupported
            # candidates in strict ablation mode and low-confidence candidates
            # otherwise.
            for original in batch:
                key = (
                    str(original.get("subject", "")).strip().lower(),
                    str(original.get("relation", "")).strip().lower(),
                    str(original.get("object", "")).strip().lower(),
                )
                linked_triple = linked_by_key.get(key)
                if linked_triple is None:
                    linked_triple = {
                        "triple": {
                            "subject": original.get("subject", ""),
                            "relation": original.get("relation", ""),
                            "object": original.get("object", ""),
                        },
                        "evidence_sentences": [],
                        "evidence_type": "none",
                        "evidence_strength": 0.0,
                        "contradictions": [],
                        "link_missing": True,
                    }
                merged = {**original, **linked_triple}
                all_linked.append(merged)
        
        return all_linked

    def _enforce_strict_source_support(
        self,
        triples: List[Dict[str, Any]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Mark support only when a quoted evidence span appears in the source.

        This is intentionally stricter than normal pipeline behavior and is
        used for the EvidenceLinker/VerificationAgent ablation. It prevents
        implicit domain knowledge or hallucinated evidence strings from being
        counted as source support.
        """
        source_norm = self._normalize_quote(text)
        for triple in triples:
            sentences = triple.get("evidence_sentences") or []
            if isinstance(sentences, str):
                sentences = [sentences]
            exact = [
                sentence
                for sentence in sentences
                if sentence and self._normalize_quote(sentence) in source_norm
            ]
            source_supported = bool(exact)
            triple["evidence_sentences"] = exact
            triple["source_supported"] = source_supported
            triple["evidence_type"] = "explicit" if source_supported else "none"
            triple["evidence_strength"] = max(
                float(triple.get("evidence_strength") or 0.0),
                0.95 if source_supported else 0.0,
            ) if source_supported else 0.0
            triple["final_confidence"] = max(
                float(triple.get("confidence") or 0.0),
                0.95,
            ) if source_supported else 0.0
            metadata = triple.setdefault("metadata", {})
            if exact:
                metadata["evidence"] = exact[0]
                metadata["source_supported"] = True
            else:
                metadata["source_supported"] = False
        return triples

    @staticmethod
    def _normalize_quote(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _get_prior_knowledge(self) -> List[Dict[str, Any]]:
        """Get prior knowledge from memory and knowledge graph."""
        prior = []
        
        # From knowledge graph
        if self.knowledge_graph:
            triples = self.knowledge_graph.triples
            # Handle both list and dict formats
            if isinstance(triples, dict):
                triple_list = list(triples.values())[:50]
            else:
                triple_list = list(triples)[:50]
            for triple in triple_list:
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
            max_tokens=4096,
        )
        
        cross_refs = result if isinstance(result, list) else result.get("cross_references", [])
        
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
        """Calculate final confidence using weighted average (not multiplicative).

        Weighted average prevents confidence crushing:
        - 40% base extraction confidence
        - 30% evidence type score
        - 30% evidence strength
        + cross-reference and contradiction adjustments
        """
        # Start with extraction confidence
        base_confidence = triple.get("confidence", 0.7)

        # Evidence type as a score (not a multiplier)
        evidence_type = triple.get("evidence_type", "inferred")
        evidence_type_score = {
            "explicit": 0.95,
            "implicit": 0.75,
            "inferred": 0.55,
        }.get(evidence_type, 0.55)

        # Evidence strength
        evidence_strength = triple.get("evidence_strength", 0.7)

        # Weighted average of the three components
        final = (
            0.4 * base_confidence
            + 0.3 * evidence_type_score
            + 0.3 * evidence_strength
        )

        # Cross-reference adjustment
        xref_status = triple.get("cross_reference_status", "novel")
        xref_adjustment = {
            "supported": 0.10,
            "contradicted": -0.20,
            "novel": 0.0,
            "refined": 0.05,
        }.get(xref_status, 0)
        final += xref_adjustment

        # Contradiction penalty
        contradictions = triple.get("contradictions", [])
        if contradictions:
            final -= 0.05 * min(len(contradictions), 3)

        return max(0.0, min(1.0, final))

    def _handle_needs_review(
        self,
        triples: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle triples that need review via deliberation."""
        # Submit to deliberation for multi-agent voting
        for triple in triples[:10]:
            triple_content = {
                "subject": triple.get("subject", triple.get("triple", {}).get("subject", "")),
                "relation": triple.get("relation", triple.get("triple", {}).get("relation", "")),
                "object": triple.get("object", triple.get("triple", {}).get("object", "")),
                "evidence_sentences": triple.get("evidence_sentences", []),
                "evidence_type": triple.get("evidence_type", "unknown"),
            }
            self.submit_for_deliberation(
                hypothesis_type="triple",
                content=triple_content,
                confidence=triple.get("final_confidence", 0.5),
                evidence=triple.get("evidence_sentences", []),
                document_id=context.document_id,
            )
        
        # Escalate
        self.escalate_to_coordinator(
            reason="Triples with weak evidence submitted for deliberation",
            items=triples,
            context={
                "document_id": context.document_id,
            },
        )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        EvidenceLinker's logic for voting on hypotheses.
        
        As the evidence expert, we vote on:
        - entity: Check if there's evidence for this entity
        - triple: Check if there's evidence for this relation
        - relation_type: Abstain (not our specialty)
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        if hypothesis_type == "entity":
            return self._vote_on_entity_evidence(hypothesis_content, context)
        elif hypothesis_type == "triple" or hypothesis_type == "relation":
            return self._vote_on_triple_evidence(hypothesis_content, context)
        elif hypothesis_type == "relation_type":
            return VoteType.ABSTAIN, 0.5, "EvidenceLinker focuses on evidence quality"
        
        return VoteType.ABSTAIN, 0.5, "EvidenceLinker cannot evaluate this hypothesis type"

    def _vote_on_entity_evidence(
        self,
        entity: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on entity based on evidence presence."""
        from multi_agent_kg.core.deliberation import VoteType
        
        entity_text = entity.get("text", "")
        source_segment = entity.get("source_segment", "")
        
        # If we have evidence of where this entity came from
        if source_segment:
            return VoteType.WEAK_ACCEPT, 0.7, "Entity has source segment evidence"
        
        # Check context if available
        if context and context.text:
            if entity_text in context.text:
                return VoteType.ACCEPT, 0.8, "Entity text found in source document"
            else:
                return VoteType.WEAK_REJECT, 0.6, "Entity text not found in document"
        
        return VoteType.ABSTAIN, 0.5, "Insufficient evidence information"

    def _vote_on_triple_evidence(
        self,
        triple: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on triple based on evidence quality, consulting memory."""
        from multi_agent_kg.core.deliberation import VoteType

        evidence_sentences = triple.get("evidence_sentences", [])
        evidence_type = triple.get("evidence_type", "unknown")
        evidence_strength = triple.get("evidence_strength", 0.5)
        contradictions = triple.get("contradictions", [])

        # Strong reject if contradictions exist
        if contradictions:
            return VoteType.REJECT, 0.8, f"Evidence contradicted: {contradictions[0][:50]}..."

        # Check SharedMemory for previously linked evidence matching this triple
        subject = triple.get("subject", "")
        obj = triple.get("object", "")
        if self.shared_memory and (subject or obj):
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            for mem in memories:
                for linked in mem.content.get("evidence_linked_triples", []):
                    linked_subj = linked.get("subject", linked.get("triple", {}).get("subject", ""))
                    linked_obj = linked.get("object", linked.get("triple", {}).get("object", ""))
                    linked_etype = linked.get("evidence_type", "")
                    if ((subject and subject.lower() == str(linked_subj).lower()) or
                            (obj and obj.lower() == str(linked_obj).lower())):
                        if linked_etype == "explicit":
                            return VoteType.STRONG_ACCEPT, 0.9, "Explicit evidence found in memory for related triple"
                        return VoteType.ACCEPT, 0.8, "Related evidence found in memory"

        # Vote based on evidence type and strength
        if evidence_type == "explicit" and evidence_strength > 0.7:
            return VoteType.STRONG_ACCEPT, 0.9, "Strong explicit evidence"
        elif evidence_type == "explicit":
            return VoteType.ACCEPT, 0.8, "Explicit evidence present"
        elif evidence_type == "implicit" and evidence_strength > 0.6:
            return VoteType.WEAK_ACCEPT, 0.7, "Implicit evidence present"
        elif evidence_type == "inferred":
            return VoteType.WEAK_ACCEPT, 0.6, "Inferred evidence (acceptable)"

        # Check if evidence sentences exist
        if evidence_sentences:
            return VoteType.WEAK_ACCEPT, 0.6, f"Has {len(evidence_sentences)} evidence sentences"

        return VoteType.WEAK_REJECT, 0.6, "No clear evidence for triple"

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
