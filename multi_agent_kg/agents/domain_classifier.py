"""
Domain Classifier Agent.

Responsible for:
- Classifying document domain (scientific, legal, general, etc.)
- Selecting appropriate extraction prompts per domain
- Adapting entity/relation types based on domain
- Storing domain context for downstream agents

This agent helps customize extraction for different content types.
"""

from typing import Any, Dict, List, Optional

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
from multi_agent_kg.core.communication import MessageBus
from multi_agent_kg.core.config import LLMConfig


# Domain configurations with expected entity and relation types
DOMAIN_CONFIGS = {
    "scientific": {
        "entity_types": [
            "CONCEPT", "THEORY", "METHOD", "DATASET", "METRIC", 
            "RESEARCHER", "INSTITUTION", "PUBLICATION", "FINDING"
        ],
        "relation_types": [
            "proposes", "evaluates", "improves", "uses", "cites",
            "achieves", "introduces", "compares", "extends", "applies"
        ],
        "focus": "technical concepts, methodologies, and experimental findings",
    },
    "biomedical": {
        "entity_types": [
            "DISEASE", "DRUG", "GENE", "PROTEIN", "CELL_TYPE",
            "ORGANISM", "PATHWAY", "SYMPTOM", "TREATMENT", "BIOMARKER"
        ],
        "relation_types": [
            "treats", "causes", "inhibits", "activates", "regulates",
            "binds_to", "expressed_in", "associated_with", "targets", "metabolizes"
        ],
        "focus": "biological entities and their interactions",
    },
    "legal": {
        "entity_types": [
            "PERSON", "ORGANIZATION", "LAW", "COURT", "JURISDICTION",
            "CONTRACT", "CASE", "STATUTE", "REGULATION", "RIGHT"
        ],
        "relation_types": [
            "governs", "applies_to", "supersedes", "amends", "violates",
            "establishes", "defines", "requires", "permits", "prohibits"
        ],
        "focus": "legal entities, regulations, and jurisdictional relationships",
    },
    "news": {
        "entity_types": [
            "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE",
            "PRODUCT", "MONEY", "PERCENTAGE", "TITLE", "NATIONALITY"
        ],
        "relation_types": [
            "announced", "acquired", "appointed", "located_in", "founded",
            "works_for", "member_of", "occurred_on", "valued_at", "affects"
        ],
        "focus": "current events, people, organizations, and their relationships",
    },
    "technical": {
        "entity_types": [
            "SYSTEM", "COMPONENT", "TECHNOLOGY", "PROTOCOL", "STANDARD",
            "VERSION", "PLATFORM", "API", "FEATURE", "REQUIREMENT"
        ],
        "relation_types": [
            "implements", "depends_on", "integrates_with", "extends",
            "replaces", "requires", "supports", "configures", "exposes", "consumes"
        ],
        "focus": "technical systems, components, and their dependencies",
    },
    "general": {
        "entity_types": [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT",
            "PRODUCT", "DATE", "QUANTITY", "ATTRIBUTE", "ACTION"
        ],
        "relation_types": [
            "is_a", "has", "located_in", "works_for", "related_to",
            "causes", "enables", "requires", "produces", "affects"
        ],
        "focus": "general entities and their relationships",
    },
}


CLASSIFICATION_PROMPT = """Classify the following text into one of these domains:
- scientific: Academic papers, research articles, technical studies
- biomedical: Medical, biological, pharmaceutical content
- legal: Laws, contracts, court cases, regulations
- news: News articles, current events, journalism
- technical: Software documentation, system specifications, APIs
- general: General purpose text that doesn't fit other categories

Analyze the text and return your classification.

TEXT:
{text}

Respond with JSON:
{{
    "domain": "<domain_name>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>",
    "key_indicators": ["<indicator1>", "<indicator2>", ...]
}}"""


class DomainClassifier(BaseAgent):
    """
    Domain Classifier Agent - Classifies document domain for tailored extraction.
    
    Responsibilities:
    1. Analyze text to determine domain
    2. Provide domain-specific entity and relation type hints
    3. Store domain classification in shared memory
    4. Escalate ambiguous classifications to coordinator
    
    Uses SharedMemory to:
    - Store domain classification for reuse
    - Track domain distribution across documents
    
    Uses MessageBus to:
    - Inform downstream agents of domain
    - Escalate ambiguous cases
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        domain_configs: Optional[Dict[str, Dict]] = None,
        confidence_threshold: float = 0.7,
    ):
        super().__init__(
            name="DomainClassifier",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.SMALL,  # Classification is simple
            quality_threshold=confidence_threshold,
        )
        self.domain_configs = domain_configs or DOMAIN_CONFIGS

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Classify document domain.
        
        Args:
            context: Processing context
            segments: Document segments from DocumentProcessor
            
        Returns:
            ExtractionResult with domain classification
        """
        self.stats["calls"] += 1
        
        # Get text for classification (use first few segments or full text)
        text_for_classification = self._get_classification_text(context, segments)
        
        # Use self-consistency for confident classification
        classification, confidence = self._classify_with_consistency(
            text_for_classification
        )
        
        # Get domain config
        domain = classification.get("domain", "general")
        domain_config = self.domain_configs.get(domain, self.domain_configs["general"])
        
        # Check if escalation needed
        needs_escalation = self.should_escalate(confidence)
        
        if needs_escalation:
            self.log(f"Low confidence ({confidence:.2f}) - escalating domain classification")
            self.escalate_to_coordinator(
                reason="Ambiguous domain classification",
                items=[classification],
                context={"text_sample": text_for_classification[:500]},
            )
        
        # Store in memory
        result_item = {
            "domain": domain,
            "confidence": confidence,
            "reasoning": classification.get("reasoning", ""),
            "key_indicators": classification.get("key_indicators", []),
            "entity_types": domain_config["entity_types"],
            "relation_types": domain_config["relation_types"],
            "focus": domain_config["focus"],
        }
        
        if self.shared_memory:
            self._store_classification(result_item, context.document_id)
        
        # Broadcast to other agents
        if self.message_bus:
            self._notify_agents(result_item, context.document_id)
        
        self.log(f"Classified as '{domain}' with confidence {confidence:.2f}")
        
        return ExtractionResult(
            items=[result_item],
            confidence=confidence,
            evidence=classification.get("key_indicators", []),
            metadata={
                "document_id": context.document_id,
            },
            needs_escalation=needs_escalation,
            escalation_reason="Ambiguous domain" if needs_escalation else None,
        )

    def _get_classification_text(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Get text for classification (first 2000 chars)."""
        if segments:
            # Combine first few segments
            texts = [s.get("text", "") for s in segments[:3]]
            return " ".join(texts)[:2000]
        
        return context.text[:2000] if context.text else ""

    def _classify_with_consistency(
        self,
        text: str,
    ) -> tuple:
        """Classify domain using self-consistency."""
        if not text:
            return {"domain": "general", "confidence": 0.5}, 0.5
        
        prompt = CLASSIFICATION_PROMPT.format(text=text)
        
        classification, confidence = self.call_llm_with_self_consistency(
            prompt=prompt,
            system_prompt="You are a domain classification expert. Classify text accurately.",
            tier=ModelTier.SMALL,
            n_samples=3,
            temperature=0.3,
        )
        
        # Ensure valid domain
        if not classification or classification.get("domain") not in self.domain_configs:
            classification = {"domain": "general", "confidence": 0.5}
            confidence = 0.5
        
        return classification, confidence

    def _store_classification(
        self,
        classification: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Store domain classification in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "classification": classification,
                "document_id": document_id,
            },
        )

    def _notify_agents(
        self,
        classification: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Notify downstream agents of domain classification."""
        from multi_agent_kg.core.communication import CommunicationType
        
        self.send_message(
            receiver="EntityExtractor",
            comm_type=CommunicationType.INFORM,
            content={
                "domain": classification["domain"],
                "entity_types": classification["entity_types"],
                "document_id": document_id,
            },
        )
        
        self.send_message(
            receiver="RelationExtractor", 
            comm_type=CommunicationType.INFORM,
            content={
                "domain": classification["domain"],
                "relation_types": classification["relation_types"],
                "document_id": document_id,
            },
        )

    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for a specific domain."""
        return self.domain_configs.get(domain, self.domain_configs["general"])
