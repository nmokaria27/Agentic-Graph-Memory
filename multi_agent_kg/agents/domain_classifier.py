"""
Domain Classifier Agent.

Analyzes raw document content to classify its domain and generate extraction 
guidance (entity types, relation types, examples) for downstream agents.

This agent dynamically identifies:
- Primary domain and hierarchical sub-domains
- Priority entity types relevant to the content
- Priority relation types relevant to the content
- Domain-specific few-shot examples for Entity Extractor
- Extraction depth and complexity parameters

Unlike a static domain classifier, this agent generates domain-specific schemas
dynamically based on the actual content, without relying on predefined categories.
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


# Prompt for domain classification and schema generation
DOMAIN_ANALYSIS_PROMPT = """Analyze the following document and DISCOVER its domain and extraction schema FROM SCRATCH.

Your task is to:
1. Identify the PRIMARY DOMAIN by analyzing the actual content (invent a specific domain name that describes THIS document)
2. Identify HIERARCHICAL SUB-DOMAINS based on the topics and themes present
3. DISCOVER what ENTITY TYPES exist in this text - look at what concepts are being discussed
4. DISCOVER what RELATION TYPES connect these entities - what relationships are described?
5. Provide 2-3 FEW-SHOT EXAMPLES of actual entities you found in the text
6. Assess the COMPLEXITY and DENSITY of extractable knowledge

CRITICAL INSTRUCTIONS:
- DO NOT use predefined categories or standard taxonomies
- CREATE entity and relation types specifically for THIS content
- Look at ACTUAL TOPICS and THEMES, not just keywords
- Entity types should capture the KEY CONCEPTS discussed in this specific document
- Relation types should capture the KEY RELATIONSHIPS described in this text
- Be as specific as possible - avoid generic types like "ENTITY" or "THING"

DOCUMENT TEXT:
{text}

Respond with JSON:
{{
    "primary_domain": "<specific domain name invented for this content>",
    "sub_domains": ["<broader domain>", "<narrower domain>", "<most specific domain>"],
    "domain_description": "<1-2 sentence description of what this document is about>",
    "confidence": <0.0-1.0>,
    "reasoning": "<why you chose this domain and these types>",
    "key_indicators": ["<key concept/theme 1>", "<key concept/theme 2>", ...],
    
    "entity_types": [
        {{
            "type": "<ENTITY_TYPE_NAME>",
            "description": "<what this represents in THIS document>",
            "priority": "<high|medium|low>",
            "examples_from_text": ["<example1>", "<example2>"]
        }}
    ],
    
    "relation_types": [
        {{
            "type": "<RELATION_TYPE_NAME>",
            "description": "<what relationship this captures in THIS context>",
            "source_types": ["<which entity types can be subjects>"],
            "target_types": ["<which entity types can be objects>"],
            "priority": "<high|medium|low>",
            "example_from_text": "<example relationship from the document>"
        }}
    ],
    
    "few_shot_examples": [
        {{
            "text_span": "<exact quote from document>",
            "entity": "<extracted entity>",
            "entity_type": "<the type you created for it>",
            "explanation": "<why this entity type fits>"
        }}
    ],
    
    "extraction_parameters": {{
        "complexity": "<low|medium|high>",
        "knowledge_density": "<sparse|moderate|dense>",
        "recommended_chunk_size": <number of tokens>,
        "requires_coreference": <true|false>,
        "has_temporal_relations": <true|false>,
        "has_hierarchical_entities": <true|false>
    }}
}}"""


# Prompt for generating relation examples
RELATION_EXAMPLES_PROMPT = """Based on the following document and the identified entity types, provide few-shot examples of relations.

DOCUMENT TEXT:
{text}

IDENTIFIED ENTITY TYPES:
{entity_types}

IDENTIFIED RELATION TYPES:
{relation_types}

For each relation type, find an example from the text. Respond with JSON:
{{
    "relation_examples": [
        {{
            "relation_type": "<RELATION_TYPE>",
            "text_span": "<exact supporting text>",
            "subject": "<subject entity>",
            "subject_type": "<entity type>",
            "object": "<object entity>",
            "object_type": "<entity type>",
            "explanation": "<why this relation exists>"
        }}
    ]
}}"""


class DomainClassifier(BaseAgent):
    """
    Domain Classifier Agent - Dynamically classifies document domain and generates
    extraction guidance tailored to the specific content.
    
    Per Spec (13B parameters):
    - Analyzes raw document content to identify primary domain
    - Detects sub-domains for hierarchical classification
    - Generates list of priority entity types relevant to the domain
    - Generates list of priority relation types relevant to the domain
    - Provides domain-specific few-shot examples for Entity Extractor
    - Sets extraction depth and complexity parameters
    - Operates in parallel with Document Processor (both receive raw input)
    
    Key Difference from Static Classifiers:
    This agent does NOT use predefined domain categories. Instead, it dynamically
    analyzes the content and generates domain-specific schemas on the fly, making
    it suitable for any domain without prior configuration.
    
    Uses SharedMemory to:
    - Store domain classification for reuse
    - Cache generated schemas for similar documents
    
    Uses MessageBus to:
    - Inform Entity Extractor of entity types and few-shot examples
    - Inform Relation Extractor of relation types and constraints
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        confidence_threshold: float = 0.7,
        max_entity_types: int = 15,
        max_relation_types: int = 15,
        use_self_consistency: bool = True,
    ):
        super().__init__(
            name="DomainClassifier",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,  # 13B per spec for nuanced classification
            quality_threshold=confidence_threshold,
        )
        self.max_entity_types = max_entity_types
        self.max_relation_types = max_relation_types
        self.use_self_consistency = use_self_consistency

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Analyze document and generate domain-specific extraction guidance.
        
        Args:
            context: Processing context with raw document
            segments: Optional document segments (uses raw text if not provided)
            
        Returns:
            ExtractionResult with domain context including:
            - primary_domain: The identified domain
            - sub_domains: Hierarchical domain path
            - entity_types: List of entity type definitions with priorities
            - relation_types: List of relation type definitions with constraints
            - few_shot_examples: Examples for entity extraction
            - relation_examples: Examples for relation extraction
            - extraction_parameters: Complexity and configuration hints
        """
        self.stats["calls"] += 1
        
        # Get representative text for analysis
        text_for_analysis = self._get_analysis_text(context, segments)
        
        if not text_for_analysis:
            self.log("No text provided for domain classification")
            return self._create_fallback_result(context.document_id)
        
        # Phase 1: Domain analysis and schema generation
        domain_analysis, confidence = self._analyze_domain(text_for_analysis)
        
        if not domain_analysis:
            self.log("Domain analysis failed, using fallback")
            return self._create_fallback_result(context.document_id)
        
        # Phase 2: Generate relation examples if we have entity and relation types
        relation_examples = []
        if domain_analysis.get("entity_types") and domain_analysis.get("relation_types"):
            relation_examples = self._generate_relation_examples(
                text_for_analysis,
                domain_analysis["entity_types"],
                domain_analysis["relation_types"],
            )
        
        # Check if escalation needed for low confidence
        needs_escalation = self.should_escalate(confidence)
        
        if needs_escalation:
            self.log(f"Low confidence ({confidence:.2f}) - escalating domain classification")
            self.escalate_to_coordinator(
                reason="Ambiguous domain classification - needs human review",
                items=[domain_analysis],
                context={"text_sample": text_for_analysis[:500]},
            )
        
        # Build domain context result
        domain_context = self._build_domain_context(
            domain_analysis, 
            relation_examples,
            context.document_id,
        )
        
        # Store in memory for downstream agents
        if self.shared_memory:
            self._store_domain_context(domain_context, context.document_id)
        
        # Notify downstream agents
        if self.message_bus:
            self._notify_agents(domain_context, context.document_id)
        
        self.log(
            f"Classified as '{domain_context['primary_domain']}' "
            f"({' → '.join(domain_context.get('sub_domains', []))}) "
            f"with {len(domain_context['entity_types'])} entity types, "
            f"{len(domain_context['relation_types'])} relation types"
        )
        
        return ExtractionResult(
            items=[domain_context],
            confidence=confidence,
            evidence=domain_analysis.get("key_indicators", []),
            metadata={
                "document_id": context.document_id,
                "complexity": domain_analysis.get("extraction_parameters", {}).get("complexity", "medium"),
            },
            needs_escalation=needs_escalation,
            escalation_reason="Ambiguous domain" if needs_escalation else None,
        )

    def _get_analysis_text(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        Get representative text for domain analysis.
        
        Uses a larger sample (up to 4000 chars) for better classification,
        preferring the beginning and sampling from middle if document is long.
        """
        max_chars = 4000
        
        if segments:
            # Combine segments, taking from beginning, middle, and end
            texts = []
            n_segments = len(segments)
            
            if n_segments <= 5:
                # Small document - use all segments
                texts = [s.get("text", "") for s in segments]
            else:
                # Large document - sample strategically
                # First 2 segments
                texts.extend([s.get("text", "") for s in segments[:2]])
                # Middle segment
                mid_idx = n_segments // 2
                texts.append(segments[mid_idx].get("text", ""))
                # Last segment
                texts.append(segments[-1].get("text", ""))
            
            combined = "\n\n".join(texts)
            return combined[:max_chars]
        
        # Fall back to raw context text
        return context.text[:max_chars] if context.text else ""

    def _analyze_domain(self, text: str) -> tuple:
        """
        Analyze document domain and generate extraction schema.
        
        Uses self-consistency with multiple samples for reliable classification.
        """
        prompt = DOMAIN_ANALYSIS_PROMPT.format(text=text)

        system_prompt = (
            "You are an expert at discovering knowledge structures from scratch. "
            "NEVER use predefined schemas or standard taxonomies. "
            "Your task is to READ the document carefully and INVENT a custom schema that fits THIS content. "
            "Focus on WHAT IS ACTUALLY DISCUSSED, not what category you think it fits into. "
            "Create entity types that capture the KEY CONCEPTS in this text. "
            "Create relation types that capture the KEY RELATIONSHIPS in this text. "
            "Be specific and descriptive - avoid generic types. "
            "Entity and relation types should be in UPPER_SNAKE_CASE format."
        )

        if self.use_self_consistency:
            analysis, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt=system_prompt,
                tier=ModelTier.MEDIUM,
                n_samples=3,
                temperature=0.4,
            )
        else:
            analysis = self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            confidence = 0.7
        
        if analysis:
            # Validate and normalize the response
            analysis = self._normalize_analysis(analysis)
        
        return analysis, confidence

    def _normalize_analysis(self, analysis: Any) -> Dict[str, Any]:
        """Normalize and validate the domain analysis response.

        Tolerates non-dict LLM outputs (lists, scalars) by coercing them to a
        minimal valid analysis shape, so a single malformed response never
        kills an entire document.
        """
        if not isinstance(analysis, dict):
            if isinstance(analysis, list):
                analysis = {"entity_types": analysis} if all(isinstance(x, (str, dict)) for x in analysis) else {}
            else:
                analysis = {}

        # Ensure required fields exist
        if "primary_domain" not in analysis:
            analysis["primary_domain"] = "General"

        if "sub_domains" not in analysis:
            analysis["sub_domains"] = [analysis["primary_domain"]]
        
        # Normalize entity types
        entity_types = analysis.get("entity_types", [])
        if isinstance(entity_types, list):
            normalized_entities = []
            for et in entity_types[:self.max_entity_types]:
                if isinstance(et, str):
                    # Simple string format - convert to dict
                    normalized_entities.append({
                        "type": et.upper().replace(" ", "_"),
                        "description": f"Entity of type {et}",
                        "priority": "medium",
                    })
                elif isinstance(et, dict):
                    et["type"] = et.get("type", "ENTITY").upper().replace(" ", "_")
                    et["priority"] = et.get("priority", "medium")
                    normalized_entities.append(et)
            analysis["entity_types"] = normalized_entities
        else:
            analysis["entity_types"] = []
        
        # Normalize relation types
        relation_types = analysis.get("relation_types", [])
        if isinstance(relation_types, list):
            normalized_relations = []
            for rt in relation_types[:self.max_relation_types]:
                if isinstance(rt, str):
                    normalized_relations.append({
                        "type": rt.upper().replace(" ", "_"),
                        "description": f"Relation of type {rt}",
                        "source_types": [],
                        "target_types": [],
                        "priority": "medium",
                    })
                elif isinstance(rt, dict):
                    rt["type"] = rt.get("type", "RELATED_TO").upper().replace(" ", "_")
                    rt["priority"] = rt.get("priority", "medium")
                    rt["source_types"] = rt.get("source_types", [])
                    rt["target_types"] = rt.get("target_types", [])
                    normalized_relations.append(rt)
            analysis["relation_types"] = normalized_relations
        else:
            analysis["relation_types"] = []
        
        # Ensure extraction parameters exist
        if "extraction_parameters" not in analysis:
            analysis["extraction_parameters"] = {
                "complexity": "medium",
                "knowledge_density": "moderate",
                "recommended_chunk_size": 512,
                "requires_coreference": True,
                "has_temporal_relations": False,
                "has_hierarchical_entities": False,
            }
        
        # Ensure few-shot examples exist
        if "few_shot_examples" not in analysis:
            analysis["few_shot_examples"] = []
        
        return analysis

    def _generate_relation_examples(
        self,
        text: str,
        entity_types: List[Dict[str, Any]],
        relation_types: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate few-shot examples for relation extraction."""
        # Format entity and relation types for prompt
        entity_str = "\n".join([
            f"- {et['type']}: {et.get('description', '')}"
            for et in entity_types
        ])
        relation_str = "\n".join([
            f"- {rt['type']}: {rt.get('description', '')}"
            for rt in relation_types
        ])
        
        prompt = RELATION_EXAMPLES_PROMPT.format(
            text=text[:2000],  # Use shorter text for examples
            entity_types=entity_str,
            relation_types=relation_str,
        )
        
        response = self.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are an expert at identifying relationships between entities in text. "
                "Provide clear, accurate examples from the given text."
            ),
            tier=ModelTier.MEDIUM,
            temperature=0.2,
            max_tokens=4096,
        )
        
        if response and "relation_examples" in response:
            return response["relation_examples"]
        
        return []

    def _build_domain_context(
        self,
        analysis: Dict[str, Any],
        relation_examples: List[Dict[str, Any]],
        document_id: str,
    ) -> Dict[str, Any]:
        """Build the complete domain context for downstream agents."""
        # Extract simple entity type names for easy consumption
        entity_type_names = [et["type"] for et in analysis.get("entity_types", [])]
        relation_type_names = [rt["type"] for rt in analysis.get("relation_types", [])]
        
        return {
            # Domain classification
            "primary_domain": analysis.get("primary_domain", "General"),
            "sub_domains": analysis.get("sub_domains", []),
            "domain_description": analysis.get("domain_description", ""),
            "confidence": analysis.get("confidence", 0.5),
            "reasoning": analysis.get("reasoning", ""),
            "key_indicators": analysis.get("key_indicators", []),
            
            # Schema for extraction - detailed format
            "entity_types": analysis.get("entity_types", []),
            "relation_types": analysis.get("relation_types", []),
            
            # Simple lists for quick access
            "entity_type_names": entity_type_names,
            "relation_type_names": relation_type_names,
            
            # Few-shot examples
            "entity_examples": analysis.get("few_shot_examples", []),
            "relation_examples": relation_examples,
            
            # Extraction parameters
            "extraction_parameters": analysis.get("extraction_parameters", {}),
            
            # Metadata
            "document_id": document_id,
        }

    def _create_fallback_result(self, document_id: str) -> ExtractionResult:
        """Create a minimal fallback result when classification fails."""
        fallback_context = {
            "primary_domain": "General",
            "sub_domains": ["General"],
            "domain_description": "Unable to classify domain - using general extraction",
            "confidence": 0.3,
            "reasoning": "Classification failed, falling back to general schema",
            "key_indicators": [],
            "entity_types": [
                {"type": "PERSON", "description": "A person or individual", "priority": "high"},
                {"type": "ORGANIZATION", "description": "An organization or institution", "priority": "high"},
                {"type": "LOCATION", "description": "A place or location", "priority": "high"},
                {"type": "CONCEPT", "description": "An abstract concept or idea", "priority": "medium"},
                {"type": "EVENT", "description": "An event or occurrence", "priority": "medium"},
                {"type": "DATE", "description": "A date or time reference", "priority": "medium"},
            ],
            "relation_types": [
                {"type": "RELATED_TO", "description": "General relationship", "source_types": [], "target_types": [], "priority": "high"},
                {"type": "LOCATED_IN", "description": "Location relationship", "source_types": [], "target_types": ["LOCATION"], "priority": "medium"},
                {"type": "PART_OF", "description": "Part-whole relationship", "source_types": [], "target_types": [], "priority": "medium"},
                {"type": "WORKS_FOR", "description": "Employment relationship", "source_types": ["PERSON"], "target_types": ["ORGANIZATION"], "priority": "medium"},
            ],
            "entity_type_names": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "DATE"],
            "relation_type_names": ["RELATED_TO", "LOCATED_IN", "PART_OF", "WORKS_FOR"],
            "entity_examples": [],
            "relation_examples": [],
            "extraction_parameters": {
                "complexity": "medium",
                "knowledge_density": "moderate",
                "recommended_chunk_size": 512,
                "requires_coreference": True,
                "has_temporal_relations": False,
                "has_hierarchical_entities": False,
            },
            "document_id": document_id,
        }
        
        return ExtractionResult(
            items=[fallback_context],
            confidence=0.3,
            evidence=[],
            metadata={"document_id": document_id, "fallback": True},
            needs_escalation=True,
            escalation_reason="Classification failed - using fallback schema",
        )

    def build_fixed_schema(
        self,
        schema_override: Dict[str, Any],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Build a domain context from a fixed schema override instead of
        dynamic discovery. Used for benchmark evaluation with known schemas.

        Args:
            schema_override: Dict with "entity_types" and "relation_types" lists.
                Each type is {"type": "TYPE_NAME", "description": "..."}.
            document_id: Document identifier.

        Returns:
            Domain context dict in the same format as dynamic classification.
        """
        entity_types = schema_override.get("entity_types", [])
        relation_types = schema_override.get("relation_types", [])

        # Normalize to expected format — preserve original casing for fixed schemas
        normalized_ents = []
        for et in entity_types:
            if isinstance(et, str):
                normalized_ents.append({
                    "type": et,
                    "description": f"Entity of type {et}",
                    "priority": "high",
                })
            elif isinstance(et, dict):
                normalized_ents.append({
                    "type": et.get("type", "ENTITY"),
                    "description": et.get("description", ""),
                    "priority": et.get("priority", "high"),
                    "examples_from_text": et.get("examples_from_text", []),
                })

        normalized_rels = []
        for rt in relation_types:
            if isinstance(rt, str):
                normalized_rels.append({
                    "type": rt,
                    "description": f"Relation of type {rt}",
                    "source_types": [],
                    "target_types": [],
                    "priority": "high",
                })
            elif isinstance(rt, dict):
                normalized_rels.append({
                    "type": rt.get("type", "RELATED_TO"),
                    "description": rt.get("description", ""),
                    "source_types": rt.get("source_types", []),
                    "target_types": rt.get("target_types", []),
                    "priority": rt.get("priority", "high"),
                })

        domain_context = {
            "primary_domain": schema_override.get("domain", "FixedSchema"),
            "sub_domains": schema_override.get("sub_domains", ["FixedSchema"]),
            "domain_description": schema_override.get("description", "Fixed schema for benchmark evaluation"),
            "schema_source": "fixed_schema_override",
            "confidence": 0.95,
            "reasoning": "Using fixed schema override",
            "key_indicators": [],
            "entity_types": normalized_ents,
            "relation_types": normalized_rels,
            "entity_type_names": [et["type"] for et in normalized_ents],
            "relation_type_names": [rt["type"] for rt in normalized_rels],
            "entity_examples": schema_override.get("entity_examples", []),
            "relation_examples": schema_override.get("relation_examples", []),
            "extraction_parameters": {
                "complexity": "medium",
                "knowledge_density": "moderate",
                "recommended_chunk_size": 512,
                "requires_coreference": True,
                "has_temporal_relations": False,
                "has_hierarchical_entities": False,
            },
            "document_id": document_id,
        }

        self.log(
            f"Fixed schema: {len(normalized_ents)} entity types, "
            f"{len(normalized_rels)} relation types"
        )

        return domain_context

    def _store_domain_context(
        self,
        domain_context: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Store domain context in shared memory for downstream agents."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "domain_context": domain_context,
                "document_id": document_id,
            },
        )

    def _notify_agents(
        self,
        domain_context: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Notify downstream agents of domain classification and schema."""
        from multi_agent_kg.core.communication import CommunicationType
        
        # Notify Entity Extractor with entity types and examples
        self.send_message(
            receiver="EntityExtractor",
            comm_type=CommunicationType.INFORM,
            content={
                "domain": domain_context["primary_domain"],
                "sub_domains": domain_context["sub_domains"],
                "entity_types": domain_context["entity_types"],
                "entity_type_names": domain_context["entity_type_names"],
                "few_shot_examples": domain_context["entity_examples"],
                "extraction_parameters": domain_context["extraction_parameters"],
                "document_id": document_id,
            },
        )
        
        # Notify Relation Extractor with relation types and constraints
        self.send_message(
            receiver="RelationExtractor",
            comm_type=CommunicationType.INFORM,
            content={
                "domain": domain_context["primary_domain"],
                "relation_types": domain_context["relation_types"],
                "relation_type_names": domain_context["relation_type_names"],
                "relation_examples": domain_context["relation_examples"],
                "entity_type_names": domain_context["entity_type_names"],
                "extraction_parameters": domain_context["extraction_parameters"],
                "document_id": document_id,
            },
        )
