"""
Extraction Verification Agent (Coordinator).

Responsible for:
- Verifying triples against source text
- Checking factual consistency
- Final quality gate before knowledge graph integration
- Cross-document verification

This is the second coordinator - the final verification step.
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
from multi_agent_kg.core.communication import MessageBus, CommunicationType, MessagePriority
from multi_agent_kg.core.config import LLMConfig


VERIFICATION_PROMPT = """Verify these triples are factually correct based on the source text.

SOURCE TEXT:
{text}

TRIPLES TO VERIFY:
{triples_json}

For each triple, verify:
1. Is the subject correctly identified?
2. Is the relation accurately stated?
3. Is the object correctly identified?
4. Is the triple supported by the text (not hallucinated)?
5. Is the confidence score appropriate?

Return:
{{
    "verified_triples": [
        {{
            "subject": "<subject>",
            "relation": "<relation>",
            "object": "<object>",
            "verified": <true/false>,
            "verification_status": "<verified|partial|rejected|hallucinated>",
            "final_confidence": <0.0-1.0>,
            "supporting_evidence": "<exact quote from text>",
            "rejection_reason": "<reason if rejected>"
        }}
    ],
    "verification_summary": {{
        "total": <count>,
        "verified": <count>,
        "partial": <count>,
        "rejected": <count>,
        "hallucinated": <count>
    }}
}}"""


CROSS_DOC_VERIFICATION_PROMPT = """Verify consistency of these triples with prior knowledge.

NEW TRIPLES:
{new_triples}

EXISTING KNOWLEDGE:
{existing_knowledge}

Check for:
1. Direct contradictions with existing facts
2. Logical inconsistencies
3. Redundancies (same fact already exists)
4. Refinements (updates to existing facts)

Return:
{{
    "consistency_results": [
        {{
            "triple": {{
                "subject": "<subject>",
                "relation": "<relation>",
                "object": "<object>"
            }},
            "consistency_status": "<consistent|contradicts|refines|redundant>",
            "related_existing": ["<related existing fact>", ...],
            "action": "<add|update|reject|merge>",
            "notes": "<explanation>"
        }}
    ]
}}"""


class ExtractionVerificationAgent(BaseAgent):
    """
    Extraction Verification Agent - Final verification before KG integration.
    
    Responsibilities:
    1. Verify triples against source text (anti-hallucination)
    2. Check cross-document consistency
    3. Final quality gate (reject if below threshold)
    4. Approve triples for knowledge graph integration
    
    Uses SharedMemory to:
    - Access prior extractions for consistency
    - Store verification results
    
    Uses MessageBus to:
    - Receive validated extractions
    - Send approved triples to KnowledgeOrganizer
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        strict_mode: bool = True,
    ):
        super().__init__(
            name="ExtractionVerificationAgent",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.LARGE,  # Verification needs high accuracy
            quality_threshold=quality_threshold,
        )
        self.strict_mode = strict_mode

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Verify extractions before KG integration.
        
        Args:
            context: Processing context
            entities: Validated entities
            triples: Validated triples
            
        Returns:
            ExtractionResult with verified extractions
        """
        self.stats["calls"] += 1
        
        entities = entities or context.entities or []
        triples = triples or context.relations or []
        
        # Process incoming messages
        messages = self.receive_messages()
        for msg in messages:
            if msg.comm_type == CommunicationType.DELEGATE and msg.content.get("action") == "verify":
                entities = msg.content.get("entities", entities)
                triples = msg.content.get("triples", triples)
        
        if not triples:
            return ExtractionResult(
                items={"entities": entities, "triples": []},
                confidence=1.0,
                metadata={"status": "no_triples_to_verify"},
            )
        
        # Step 1: Verify against source text
        verification_result = self._verify_against_source(
            context.text,
            triples,
        )
        
        # Categorize results
        verified = []
        partial = []
        rejected = []
        
        for v in verification_result.get("verified_triples", []):
            status = v.get("verification_status", "rejected")
            if status == "verified":
                verified.append(v)
            elif status == "partial":
                if not self.strict_mode:
                    partial.append(v)
                else:
                    rejected.append(v)
            else:
                rejected.append(v)
        
        # Step 2: Cross-document consistency check
        existing_knowledge = self._get_existing_knowledge()
        if existing_knowledge and verified:
            verified = self._check_cross_doc_consistency(verified, existing_knowledge)
        
        # Filter by final confidence threshold
        approved = []
        for triple in verified + partial:
            if triple.get("final_confidence", 0) >= self.quality_threshold:
                approved.append(triple)
            else:
                rejected.append(triple)
        
        # Store verification results
        if self.shared_memory:
            self._store_verification_results(
                approved,
                rejected,
                context.document_id,
            )
        
        # Forward to KnowledgeOrganizer
        if self.message_bus:
            self._forward_to_organizer(
                entities,
                approved,
                context.document_id,
            )
        
        summary = verification_result.get("verification_summary", {})
        self.log(
            f"Verified: {len(verified)}, Approved: {len(approved)}, "
            f"Rejected: {len(rejected)}"
        )
        
        return ExtractionResult(
            items={
                "entities": entities,
                "approved_triples": approved,
                "rejected_triples": rejected,
            },
            confidence=len(approved) / len(triples) if triples else 1.0,
            metadata={
                "document_id": context.document_id,
                "verification_summary": summary,
                "approved_count": len(approved),
                "rejected_count": len(rejected),
            },
        )

    def _verify_against_source(
        self,
        text: str,
        triples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify triples against source text."""
        if not triples:
            return {"verified_triples": [], "verification_summary": {}}
        
        triples_json = json.dumps([
            {
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
                "confidence": t.get("confidence", 0.7),
            }
            for t in triples[:30]
        ], indent=2)
        
        prompt = VERIFICATION_PROMPT.format(
            text=text[:5000],
            triples_json=triples_json,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are an expert fact verifier. Be strict about accuracy. "
                "Reject any triple that is not clearly supported by the text."
            ),
            tier=ModelTier.LARGE,
            max_tokens=4096,
        )
        
        return result

    def _get_existing_knowledge(self) -> List[Dict[str, Any]]:
        """Get existing knowledge for consistency check."""
        existing = []
        
        # From knowledge graph
        if self.knowledge_graph:
            for triple_id, triple in list(self.knowledge_graph.triples.items())[:100]:
                existing.append({
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                })
        
        return existing

    def _check_cross_doc_consistency(
        self,
        triples: List[Dict[str, Any]],
        existing: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check cross-document consistency."""
        if not triples or not existing:
            return triples
        
        new_triples_json = json.dumps([
            {
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": t.get("object", ""),
            }
            for t in triples[:20]
        ], indent=2)
        
        existing_json = json.dumps(existing[:50], indent=2)
        
        prompt = CROSS_DOC_VERIFICATION_PROMPT.format(
            new_triples=new_triples_json,
            existing_knowledge=existing_json,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at knowledge consistency checking. Be thorough.",
            tier=ModelTier.LARGE,
            max_tokens=4096,
        )
        
        # Apply consistency results
        consistency_results = result.get("consistency_results", [])
        
        filtered = []
        for triple in triples:
            # Find matching result
            matching = None
            for cr in consistency_results:
                cr_triple = cr.get("triple", {})
                if (cr_triple.get("subject") == triple.get("subject") and
                    cr_triple.get("relation") == triple.get("relation") and
                    cr_triple.get("object") == triple.get("object")):
                    matching = cr
                    break
            
            if matching:
                action = matching.get("action", "add")
                if action in ["add", "update", "merge"]:
                    triple["consistency_status"] = matching.get("consistency_status")
                    triple["consistency_action"] = action
                    filtered.append(triple)
                # Reject if action is "reject"
            else:
                filtered.append(triple)
        
        return filtered

    def _store_verification_results(
        self,
        approved: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store verification results in memory."""
        self.store_in_memory(
            memory_type=MemoryType.WORKING,
            content={
                "approved_triples": approved,
                "rejected_triples": rejected,
                "document_id": document_id,
            },
        )

    def _forward_to_organizer(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Forward approved extractions to KnowledgeOrganizer."""
        self.send_message(
            receiver="KnowledgeOrganizer",
            comm_type=CommunicationType.DELEGATE,
            content={
                "action": "integrate",
                "entities": entities,
                "triples": triples,
                "document_id": document_id,
            },
            priority=MessagePriority.NORMAL,
        )
