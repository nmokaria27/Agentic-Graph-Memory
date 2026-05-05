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
from multi_agent_kg.core.communication import MessageBus, CommunicationType, MessagePriority
from multi_agent_kg.core.config import LLMConfig


VERIFICATION_PROMPT = """Verify these triples are factually correct based on the source text.

SOURCE TEXT:
{text}

TRIPLES TO VERIFY:
{triples_json}

For each triple, verify:
1. Is the subject mentioned or reasonably implied in the text?
2. Does the relation capture a relationship that exists in the text (explicit or inferred)?
3. Is the object mentioned or reasonably implied in the text?
4. Is the triple consistent with the text's content and domain?
5. Is the confidence score reasonable?

REJECT if subject or object is:
- A single common word (e.g., "study", "result", "data", "method", "analysis")
- A generic phrase (e.g., "the study", "the authors", "the results", "these patients")
- A bare number without context (e.g., "42", "0.85")
- A pronoun or article (e.g., "it", "this", "the", "they")
- An adjective alone (e.g., "significant", "high", "low")

IMPORTANT:
- Accept BOTH explicit AND reasonably inferred relationships
- Focus on whether the triple captures real knowledge from the domain
- Accept if the triple is consistent with the text, even if not explicitly stated
- Only reject if clearly contradicted, completely unrelated, or contains garbage entities
- Partial support counts as valid - mark as "partial" with slightly lower confidence
- Inferred relationships between domain entities are valuable knowledge
- Different entity/relation types are OK if they capture the domain correctly

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
        quality_threshold: float = 0.45,  # Further lowered to accept more inferred knowledge
        strict_mode: bool = False,  # Allow partial verifications
        strict_source_only: bool = False,
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
        self.strict_source_only = strict_source_only

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
        if self.strict_source_only:
            verification_result = self._verify_by_existing_source_links(context.text, triples)
        else:
            verification_result = self._verify_against_source(
                context.text,
                triples,
            )
        
        # Categorize results
        verified = []
        partial = []
        rejected = []
        
        print("\n" + "="*70)
        print("[VERIFICATION DEBUG] Categorizing verification results...")
        print("="*70)
        
        from multi_agent_kg.utils.debug_logger import get_debug_logger
        logger = get_debug_logger()
        
        for v in verification_result.get("verified_triples", []):
            status = v.get("verification_status", "rejected")
            triple = v.get('triple', {})
            triple_str = f"{triple.get('subject', '?')} -> {triple.get('relation', '?')} -> {triple.get('object', '?')}"
            reason = v.get('verification_reasoning', v.get('rejection_reason', 'No reason provided'))
            
            if status == "verified":
                verified.append(v)
                print(f"  ✓ VERIFIED: {triple_str}")
                logger.log_decision("ExtractionVerificationAgent", "triple", triple, "VERIFIED", reason)
            elif status == "partial":
                if not self.strict_mode:
                    partial.append(v)
                    print(f"  ~ PARTIAL: {triple_str}")
                    logger.log_decision("ExtractionVerificationAgent", "triple", triple, "PARTIAL", reason)
                else:
                    rejected.append(v)
                    print(f"  ✗ REJECTED (strict mode): {triple_str}")
                    logger.log_decision("ExtractionVerificationAgent", "triple", triple, "REJECTED", f"Strict mode - {reason}")
            else:
                rejected.append(v)
                print(f"  ✗ REJECTED: {triple_str} | Reason: {reason}")
                logger.log_decision("ExtractionVerificationAgent", "triple", triple, "REJECTED", reason)
        
        # Step 2: Cross-document consistency check
        existing_knowledge = self._get_existing_knowledge()
        if existing_knowledge and verified and not self.strict_source_only:
            verified = self._check_cross_doc_consistency(verified, existing_knowledge)
        
        # Filter by final confidence threshold
        print(f"\n[VERIFICATION DEBUG] Filtering by confidence threshold ({self.quality_threshold})...")
        approved = []
        failed_conf_count = 0
        
        for triple in verified + partial:
            conf = triple.get("final_confidence", 0)
            triple_data = triple.get('triple', {})
            if not triple_data:
                triple_data = {
                    "subject": triple.get("subject", ""),
                    "relation": triple.get("relation", ""),
                    "object": triple.get("object", ""),
                }
            triple_str = f"{triple_data.get('subject', '?')} -> {triple_data.get('relation', '?')} -> {triple_data.get('object', '?')}"
            
            if conf >= self.quality_threshold:
                approved_payload = {
                    **triple,
                    **{
                        key: value
                        for key, value in triple_data.items()
                        if key in {"subject", "relation", "object"}
                    },
                }
                approved.append(approved_payload)
                print(f"  ✓ APPROVED (conf={conf:.2f}): {triple_str}")
                logger.log_decision("ExtractionVerificationAgent", "triple", triple_data, "APPROVED", 
                                  f"Confidence {conf:.2f} >= threshold {self.quality_threshold}", conf)
            else:
                rejected.append(triple)
                failed_conf_count += 1
                print(f"  ✗ REJECTED (conf={conf:.2f} < {self.quality_threshold}): {triple_str}")
                logger.log_decision("ExtractionVerificationAgent", "triple", triple_data, "REJECTED", 
                                  f"Confidence {conf:.2f} < threshold {self.quality_threshold}", conf)
        
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
        
        print(f"\n[VERIFICATION SUMMARY]")
        print(f"  Input: {len(triples)} triples")
        print(f"  Verified: {len(verified)}")
        print(f"  Partial: {len(partial)}")
        print(f"  Initially Rejected: {len(rejected) - len([t for t in verified + partial if t.get('final_confidence', 0) < self.quality_threshold])}")
        print(f"  Failed Confidence Threshold: {len([t for t in verified + partial if t.get('final_confidence', 0) < self.quality_threshold])}")
        print(f"  Final Approved: {len(approved)}")
        print(f"  Total Rejected: {len(rejected)}")
        
        summary = verification_result.get("verification_summary", {}) if isinstance(verification_result, dict) else {}
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

    def _verify_by_existing_source_links(
        self,
        text: str,
        triples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deterministic source-only verification for ablation runs.

        EvidenceLinker must already have attached source-supported evidence.
        This avoids using the verification LLM as a second implicit-knowledge
        judge when the experiment is explicitly testing evidence filtering.
        """
        source_norm = self._normalize_quote(text)
        verified: List[Dict[str, Any]] = []
        summary = {"total": len(triples), "verified": 0, "partial": 0, "rejected": 0, "hallucinated": 0}

        for triple in triples:
            evidence = triple.get("evidence_sentences") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            metadata = triple.get("metadata") if isinstance(triple.get("metadata"), dict) else {}
            if metadata.get("evidence"):
                evidence = list(evidence) + [metadata.get("evidence")]
            exact = [
                item
                for item in evidence
                if item and self._normalize_quote(item) in source_norm
            ]
            payload = {
                "subject": triple.get("subject", triple.get("triple", {}).get("subject", "")),
                "relation": triple.get("relation", triple.get("triple", {}).get("relation", "")),
                "object": triple.get("object", triple.get("triple", {}).get("object", "")),
            }
            if exact:
                summary["verified"] += 1
                verified.append(
                    {
                        "triple": payload,
                        "verified": True,
                        "verification_status": "verified",
                        "final_confidence": max(float(triple.get("confidence") or 0.0), 0.95),
                        "supporting_evidence": exact[0],
                        "rejection_reason": "",
                    }
                )
            else:
                summary["rejected"] += 1
                verified.append(
                    {
                        "triple": payload,
                        "verified": False,
                        "verification_status": "rejected",
                        "final_confidence": 0.0,
                        "supporting_evidence": "",
                        "rejection_reason": "No exact source evidence span attached by EvidenceLinker.",
                    }
                )

        return {"verified_triples": verified, "verification_summary": summary}

    @staticmethod
    def _normalize_quote(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _verify_against_source(
        self,
        text: str,
        triples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify triples against source text, in batches to respect token limits."""
        if not triples:
            return {"verified_triples": [], "verification_summary": {}}

        BATCH = 30
        all_verified: List[Dict[str, Any]] = []
        summary_totals = {"total": 0, "verified": 0, "partial": 0, "rejected": 0, "hallucinated": 0}

        for i in range(0, len(triples), BATCH):
            batch = triples[i:i + BATCH]
            triples_json = json.dumps([
                {
                    "subject": t.get("subject", ""),
                    "relation": t.get("relation", ""),
                    "object": t.get("object", ""),
                    "confidence": t.get("confidence", 0.7),
                }
                for t in batch
            ], indent=2)

            prompt = VERIFICATION_PROMPT.format(
                text=text[:6000],
                triples_json=triples_json,
            )

            result = self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are an expert fact verifier. Accept BOTH explicit and "
                    "reasonably inferred relationships. Only reject triples that "
                    "are clearly contradicted or completely unsupported by the text. "
                    "Partial support counts as valid with lower confidence."
                ),
                tier=ModelTier.LARGE,
                max_tokens=4096,
            )

            if isinstance(result, list):
                all_verified.extend(result)
                batch_summary = {}
            else:
                all_verified.extend(result.get("verified_triples", []))
                batch_summary = result.get("verification_summary", {})
            for key in summary_totals:
                summary_totals[key] += batch_summary.get(key, 0)

        return {"verified_triples": all_verified, "verification_summary": summary_totals}

    def _get_existing_knowledge(self) -> List[Dict[str, Any]]:
        """Get existing knowledge for consistency check."""
        existing = []
        
        # From knowledge graph
        if self.knowledge_graph:
            for triple in list(self.knowledge_graph.triples)[:100]:
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
            for t in triples
        ], indent=2)
        
        existing_json = json.dumps(existing, indent=2)
        
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
        consistency_results = result if isinstance(result, list) else result.get("consistency_results", [])
        
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
