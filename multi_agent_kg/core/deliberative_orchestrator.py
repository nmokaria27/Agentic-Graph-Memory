"""
Deliberative Multi-Agent Orchestrator.

This orchestrator implements the full integrated pipeline with:
- SharedMemory for cross-document context and blackboard voting
- MessageBus for inter-agent communication
- DeliberationCoordinator for multi-agent voting and debate
- Tiered model selection
- Iterative refinement with quality thresholds
- Escalation and deliberation mechanisms

Architecture:
  Workers: DocumentProcessor -> DomainClassifier -> EntityExtractor -> 
           RelationExtractor -> EvidenceLinker
  Coordinators: ExtractionValidator -> ExtractionVerificationAgent -> 
                KnowledgeOrganizer

Novel Features:
- Multi-agent deliberation with voting and debate
- Blackboard pattern for hypothesis posting
- Self-consistency for confidence estimation
- Cross-document entity resolution
- Open-world relation discovery
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CollaborationProtocol
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.deliberation import DeliberationCoordinator, VoteType

from multi_agent_kg.agents.base import AgentContext, ModelTier
from multi_agent_kg.agents.document_processor import DocumentProcessor
from multi_agent_kg.agents.domain_classifier import DomainClassifier
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.agents.evidence_linker import EvidenceLinker
from multi_agent_kg.agents.extraction_validator import ExtractionValidator
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer

try:
    from multi_agent_kg.utils.kg_visualizer import KGVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False


class DeliberativeOrchestrator:
    """
    Deliberative Multi-Agent Orchestrator for Knowledge Graph Construction.
    
    This orchestrator coordinates 8 agents in a tiered pipeline with
    full multi-agent deliberation support.
    
    Worker Agents (extraction):
    1. DocumentProcessor: Ingests and segments documents
    2. DomainClassifier: Classifies domain for tailored extraction
    3. EntityExtractor: Multi-stage entity extraction with self-consistency
    4. RelationExtractor: RHF-style relation extraction with open-world support
    5. EvidenceLinker: Links triples to source evidence
    
    Coordinator Agents (validation):
    6. ExtractionValidator: Validates, refines, and coordinates deliberation
    7. ExtractionVerificationAgent: Final verification against source
    8. KnowledgeOrganizer: Integrates into knowledge graph
    
    Novel Features:
    - DeliberationCoordinator: Multi-agent voting and debate on hypotheses
    - SharedMemory: Episodic, semantic, working memory + blackboard pattern
    - MessageBus: Inter-agent communication for escalation and feedback
    - Self-Consistency: Multiple LLM samples for confidence estimation
    - Cross-Document: Entity resolution across multiple documents
    - Open-World: Discovery of new relation types
    
    Deliberation Flow:
    1. Worker extracts with low confidence → submits hypothesis
    2. DeliberationCoordinator broadcasts vote request
    3. Other workers vote with rationales
    4. If conflict → debate phase with arguments
    5. Coordinator resolves with weighted consensus
    6. Accepted hypotheses added to final output
    
    Quality Assurance:
    - Iterative refinement up to 4 iterations
    - Quality threshold of 0.85 for acceptance
    - Multi-agent voting for ambiguous cases
    - Debate loop for conflicting votes
    - Escalation to coordinators for low-confidence items
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        quality_threshold: float = 0.60,
        max_refinement_iterations: int = 4,
        enable_self_consistency: bool = True,
        enable_open_world: bool = True,
        enable_cross_document: bool = True,
        enable_deliberation: bool = True,
        model_tiers: Optional[Dict[ModelTier, str]] = None,
        debug_logger = None,
    ):
        """
        Initialize the deliberative orchestrator.
        
        Args:
            llm_config: Base LLM configuration
            knowledge_graph: Existing KG or creates new
            quality_threshold: Minimum confidence for acceptance (default 0.60)
            max_refinement_iterations: Max refinement loops (default 4)
            enable_self_consistency: Use self-consistency for confidence
            enable_open_world: Allow discovery of new relation types
            enable_cross_document: Enable cross-document entity resolution
            enable_deliberation: Enable multi-agent voting and debate
            model_tiers: Custom model tier mapping
            debug_logger: Debug logger for tracking communications
        """
        self.llm_config = llm_config or LLMConfig()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.quality_threshold = quality_threshold
        self.max_refinement_iterations = max_refinement_iterations
        self.enable_self_consistency = enable_self_consistency
        self.enable_open_world = enable_open_world
        self.enable_cross_document = enable_cross_document
        self.enable_deliberation = enable_deliberation
        self.debug_logger = debug_logger
        
        # Model tier configuration — uses env vars or defaults
        import os
        self.model_tiers = model_tiers or {
            ModelTier.SMALL: os.getenv("OLLAMA_MODEL_SMALL", "gemma3:27b"),
            ModelTier.MEDIUM: os.getenv("OLLAMA_MODEL_MEDIUM", "gemma3:27b"),
            ModelTier.LARGE: os.getenv("OLLAMA_MODEL_LARGE", "gemma3:27b"),
        }
        
        # Shared infrastructure
        self.shared_memory = SharedMemory()
        self.message_bus = MessageBus(debug_logger=self.debug_logger)
        self.collab = CollaborationProtocol(self.message_bus)
        
        # Deliberation coordinator
        self.deliberation_coordinator = DeliberationCoordinator(
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            voting_agents=["EntityExtractor", "RelationExtractor", "EvidenceLinker"],
            consensus_threshold=0.6,
            min_votes=2,
            debug_logger=self.debug_logger,
        ) if enable_deliberation else None
        
        # Initialize agents
        self._init_agents()
        
        # Tracking
        self.document_count = 0
        self.session_start = datetime.now()
        self.processing_history = []
        
        self._print_header()

    # ── Per-agent model assignments ───────────────────────────────────────
    # These can be overridden by env vars (AGENT_MODEL_<NAME>) or by passing
    # model_tiers to the constructor.  When set, the agent ignores the global
    # tier table and always calls this specific model.
    import os as _os
    AGENT_MODELS: Dict[str, str] = {
        # Worker agents
        "DocumentProcessor":          _os.getenv("AGENT_MODEL_DOCUMENT_PROCESSOR",   "qwen3:4b"),
        "DomainClassifier":           _os.getenv("AGENT_MODEL_DOMAIN_CLASSIFIER",    "qwen3:8b"),
        "EntityExtractor":            _os.getenv("AGENT_MODEL_ENTITY_EXTRACTOR",     "qwen3:8b"),
        "RelationExtractor":          _os.getenv("AGENT_MODEL_RELATION_EXTRACTOR",   "qwen3:8b"),
        "EvidenceLinker":             _os.getenv("AGENT_MODEL_EVIDENCE_LINKER",      "qwen3:4b"),
        # Coordinator agents
        "ExtractionVerificationAgent":_os.getenv("AGENT_MODEL_VERIFICATION_AGENT",   "deepseek-r1:14b"),
        "KnowledgeOrganizer":         _os.getenv("AGENT_MODEL_KNOWLEDGE_ORGANIZER",  "gpt-oss:20b"),
        # ExtractionValidator uses the global tier table (no override by default)
    }

    def _init_agents(self) -> None:
        """Initialize all agents with shared infrastructure."""

        # Worker Agents
        self.document_processor = DocumentProcessor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        self.document_processor.model_override = self.AGENT_MODELS["DocumentProcessor"]

        self.domain_classifier = DomainClassifier(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        self.domain_classifier.model_override = self.AGENT_MODELS["DomainClassifier"]

        self.entity_extractor = EntityExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
        )
        self.entity_extractor.model_override = self.AGENT_MODELS["EntityExtractor"]

        self.relation_extractor = RelationExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
            enable_open_world=self.enable_open_world,
        )
        self.relation_extractor.model_override = self.AGENT_MODELS["RelationExtractor"]

        self.evidence_linker = EvidenceLinker(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            enable_cross_reference=self.enable_cross_document,
        )
        self.evidence_linker.model_override = self.AGENT_MODELS["EvidenceLinker"]

        # Coordinator Agents
        self.extraction_validator = ExtractionValidator(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            max_iterations=self.max_refinement_iterations,
        )
        # ExtractionValidator has no override — uses the global tier table

        self.verification_agent = ExtractionVerificationAgent(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
        )
        self.verification_agent.model_override = self.AGENT_MODELS["ExtractionVerificationAgent"]

        self.knowledge_organizer = KnowledgeOrganizer(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        self.knowledge_organizer.model_override = self.AGENT_MODELS["KnowledgeOrganizer"]

        # Visualizer will be initialized on-demand when export() is called
        self.visualizer = None

        # Set deliberation coordinator on all agents
        if self.deliberation_coordinator:
            self._setup_deliberation()

    def _setup_deliberation(self) -> None:
        """Set up deliberation coordinator for all agents."""
        all_agents = [
            self.document_processor,
            self.domain_classifier,
            self.entity_extractor,
            self.relation_extractor,
            self.evidence_linker,
            self.extraction_validator,
            self.verification_agent,
            self.knowledge_organizer,
        ]
        
        for agent in all_agents:
            agent.set_deliberation_coordinator(self.deliberation_coordinator)

    def _print_header(self) -> None:
        """Print orchestrator header."""
        print("\n" + "=" * 70)
        print("DELIBERATIVE MULTI-AGENT KNOWLEDGE GRAPH FRAMEWORK")
        print("=" * 70)
        print("Per-Agent Models:")
        agent_model_display = [
            ("DocumentProcessor",           self.AGENT_MODELS.get("DocumentProcessor", "—")),
            ("DomainClassifier",            self.AGENT_MODELS.get("DomainClassifier", "—")),
            ("EntityExtractor",             self.AGENT_MODELS.get("EntityExtractor", "—")),
            ("RelationExtractor",           self.AGENT_MODELS.get("RelationExtractor", "—")),
            ("EvidenceLinker",              self.AGENT_MODELS.get("EvidenceLinker", "—")),
            ("ExtractionValidator",         f"{self.model_tiers.get(ModelTier.LARGE, '?')} (tier-based)"),
            ("ExtractionVerificationAgent", self.AGENT_MODELS.get("ExtractionVerificationAgent", "—")),
            ("KnowledgeOrganizer",          self.AGENT_MODELS.get("KnowledgeOrganizer", "—")),
        ]
        for agent_name, model in agent_model_display:
            print(f"  {agent_name:<30} {model}")
        print(f"\nFeatures:")
        print(f"  Self-Consistency: {'Enabled' if self.enable_self_consistency else 'Disabled'}")
        print(f"  Open-World Relations: {'Enabled' if self.enable_open_world else 'Disabled'}")
        print(f"  Cross-Document Resolution: {'Enabled' if self.enable_cross_document else 'Disabled'}")
        print(f"  Multi-Agent Deliberation: {'Enabled' if self.enable_deliberation else 'Disabled'}")
        print(f"\nQuality Settings:")
        print(f"  Threshold: {self.quality_threshold}")
        print(f"  Max Refinement Iterations: {self.max_refinement_iterations}")
        if self.enable_deliberation:
            print(f"\nDeliberation Settings:")
            print(f"  Voting Agents: EntityExtractor, RelationExtractor, EvidenceLinker")
            print(f"  Consensus Threshold: 0.6")
            print(f"  Min Votes Required: 2")
        print("=" * 70 + "\n")

    def process_document(
        self,
        text: Optional[str] = None,
        source_path: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document through the complete multi-agent pipeline.
        
        Args:
            text: Document text content
            source_path: Path to document file
            document_id: Optional document ID
            metadata: Optional document metadata
            
        Returns:
            Processing results including extracted entities and triples
        """
        self.document_count += 1
        start_time = datetime.now()
        
        # Generate document ID
        if not document_id:
            content = text or source_path or ""
            doc_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            document_id = f"doc_{self.document_count}_{doc_hash}"
        
        print(f"\n{'='*70}")
        print(f"Processing Document: {document_id}")
        print(f"{'='*70}")
        
        # Create context
        context = AgentContext(
            document_id=document_id,
            text=text or "",
            quality_threshold=self.quality_threshold,
            max_iterations=self.max_refinement_iterations,
        )
        
        results = {}

        # ===== CHECK FOR SHORT TEXT → SINGLE-PASS MODE =====
        word_count = len((text or "").split())
        if word_count > 0 and word_count <= 200:
            print(f"\n  [SHORT TEXT MODE] {word_count} words — using single-pass extraction")
            return self._process_short_text(context, document_id, start_time, metadata)

        # ===== WORKER AGENTS (FULL PIPELINE) =====

        # Step 1: Document Processing
        if self.debug_logger:
            self.debug_logger.log_stage_header(1, "Document Processing")
        print("\n[1/9] Document Processing")
        print("-" * 50)
        doc_result = self.document_processor.run(context, source_path=source_path)
        segments = doc_result.items
        results["segments"] = len(segments)
        print(f"  Segments: {len(segments)}")
        
        # Step 2: Domain Classification
        if self.debug_logger:
            self.debug_logger.log_stage_header(2, "Domain Classification")
        print("\n[2/9] Domain Classification")
        print("-" * 50)
        domain_result = self.domain_classifier.run(context, segments=segments)
        domain_config = domain_result.items[0] if domain_result.items else {}
        context.domain = domain_config.get("domain", "general")
        results["domain"] = context.domain
        print(f"  Domain: {context.domain} (confidence: {domain_result.confidence:.2f})")
        
        # Step 3: Entity Extraction
        if self.debug_logger:
            self.debug_logger.log_stage_header(3, "Entity Extraction (Multi-Stage)")
        print("\n[3/9] Entity Extraction (Multi-Stage)")
        print("-" * 50)
        entity_result = self.entity_extractor.run(
            context, 
            segments=segments,
            domain_config=domain_config,
        )
        entities = entity_result.items
        context.entities = entities
        results["entities_extracted"] = len(entities)
        print(f"  Entities: {len(entities)} (confidence: {entity_result.confidence:.2f})")
        if entity_result.needs_escalation:
            print(f"  Escalation: {entity_result.escalation_reason}")
        
        # Step 4: Relation Extraction (RHF)
        if self.debug_logger:
            self.debug_logger.log_stage_header(4, "Relation Extraction (RHF Pipeline)")
        print("\n[4/9] Relation Extraction (RHF Pipeline)")
        print("-" * 50)
        relation_result = self.relation_extractor.run(
            context,
            segments=segments,
            entities=entities,
            domain_config=domain_config,
        )
        triples = relation_result.items
        context.relations = triples
        results["triples_extracted"] = len(triples)
        print(f"  Triples: {len(triples)} (confidence: {relation_result.confidence:.2f})")
        if relation_result.metadata.get("new_relations_discovered"):
            print(f"  New Relation Types: {relation_result.metadata['new_relations_discovered']}")

        # Step 4b: Inferred Triple Generation
        print("\n[4b/9] Inferred Triple Generation")
        print("-" * 50)
        inferred_triples = self.relation_extractor.generate_inferred_triples(
            text=context.text,
            entities=entities,
            triples=triples,
        )
        if inferred_triples:
            triples = triples + inferred_triples
            context.relations = triples
            results["inferred_triples"] = len(inferred_triples)
            print(f"  Inferred: {len(inferred_triples)} new triples")
        else:
            results["inferred_triples"] = 0
            print("  No additional triples inferred")
        results["triples_total"] = len(triples)
        print(f"  Total triples: {len(triples)}")

        # Step 5: Evidence Linking
        if self.debug_logger:
            self.debug_logger.log_stage_header(5, "Evidence Linking")
        print("\n[5/9] Evidence Linking")
        print("-" * 50)
        evidence_result = self.evidence_linker.run(
            context,
            triples=triples,
            segments=segments,
        )
        linked_triples = evidence_result.items
        results["triples_linked"] = len(linked_triples)
        print(f"  Linked: {len(linked_triples)} (confidence: {evidence_result.confidence:.2f})")
        
        # Step 6: Multi-Agent Deliberation
        if self.debug_logger:
            self.debug_logger.log_stage_header(6, "Multi-Agent Deliberation")
        print("\n[6/9] Multi-Agent Deliberation")
        print("-" * 50)
        
        if self.enable_deliberation and self.deliberation_coordinator:
            deliberation_results = self._run_deliberation_phase(
                context=context,
                entities=entities,
                triples=linked_triples,
                segments=segments,
            )
            results["voting_sessions"] = deliberation_results.get("voting_sessions", 0)
            results["debates_triggered"] = deliberation_results.get("debates_triggered", 0)
            results["items_accepted_by_vote"] = deliberation_results.get("accepted", 0)
            results["items_rejected_by_vote"] = deliberation_results.get("rejected", 0)
            
            # Update entities/triples based on deliberation
            if deliberation_results.get("refined_entities"):
                entities = deliberation_results["refined_entities"]
                context.entities = entities
            if deliberation_results.get("refined_triples"):
                linked_triples = deliberation_results["refined_triples"]
        else:
            results["voting_sessions"] = 0
            results["debates_triggered"] = 0
            results["items_accepted_by_vote"] = 0
            results["items_rejected_by_vote"] = 0
            print("  Deliberation disabled - skipping")
        
        print(f"  Voting Sessions: {results['voting_sessions']}")
        print(f"  Debates Triggered: {results['debates_triggered']}")
        print(f"  Accepted by Vote: {results['items_accepted_by_vote']}")
        print(f"  Rejected by Vote: {results['items_rejected_by_vote']}")
        
        # ===== COORDINATOR AGENTS =====
        
        # Step 7: Extraction Validation
        if self.debug_logger:
            self.debug_logger.log_stage_header(7, "Extraction Validation (Iterative Refinement)")
        print("\n[7/9] Extraction Validation (Iterative Refinement)")
        print("-" * 50)
        validation_result = self.extraction_validator.run(
            context,
            entities=entities,
            triples=linked_triples,
        )
        validated = validation_result.items
        results["refinement_iterations"] = validation_result.metadata.get("refinement_iterations", 0)
        print(f"  Iterations: {results['refinement_iterations']}")
        print(f"  Quality: {validation_result.confidence:.2f}")
        
        # Step 8: Verification
        if self.debug_logger:
            self.debug_logger.log_stage_header(8, "Extraction Verification")
        print("\n[8/9] Extraction Verification")
        print("-" * 50)
        verification_result = self.verification_agent.run(
            context,
            entities=validated.get("entities", entities),
            triples=validated.get("triples", linked_triples),
        )
        verified = verification_result.items
        results["approved_triples"] = len(verified.get("approved_triples", []))
        results["rejected_triples"] = len(verified.get("rejected_triples", []))
        print(f"  Approved: {results['approved_triples']}")
        print(f"  Rejected: {results['rejected_triples']}")
        
        # Step 9: Knowledge Organization
        if self.debug_logger:
            self.debug_logger.log_stage_header(9, "Knowledge Graph Integration")
        print("\n[9/9] Knowledge Graph Integration")
        print("-" * 50)
        integration_result = self.knowledge_organizer.run(
            context,
            entities=verified.get("entities", entities),
            triples=verified.get("approved_triples", []),
        )
        kg_stats = integration_result.metadata.get("kg_stats", {})
        results["kg_entities"] = kg_stats.get("total_entities", 0)
        results["kg_triples"] = kg_stats.get("total_triples", 0)
        print(f"  KG Entities: {results['kg_entities']}")
        print(f"  KG Triples: {results['kg_triples']}")
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        results["processing_time_seconds"] = elapsed
        
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Document: {document_id}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Entities: {results['entities_extracted']} extracted -> {results['kg_entities']} in KG")
        print(f"Triples: {results['triples_extracted']} extracted -> {results['approved_triples']} approved -> {results['kg_triples']} in KG")
        print(f"{'='*70}\n")
        
        # Store in history
        self.processing_history.append({
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })
        
        return results

    def _process_short_text(
        self,
        context: AgentContext,
        document_id: str,
        start_time: "datetime",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Single-pass extraction for short texts (< 200 words).

        Uses a single Chain-of-Thought LLM call that extracts entities AND
        relations together, avoiding the overhead of the full multi-stage
        pipeline.  This follows the EDC (Extract-Define-Canonicalize)
        pattern from EMNLP 2024.
        """
        from multi_agent_kg.llm.openai_client import chat_completion_json

        text = context.text
        results: Dict[str, Any] = {"segments": 1, "mode": "single_pass"}

        prompt = f"""Extract a knowledge graph from the following text.

TEXT:
{text}

INSTRUCTIONS — follow this chain of thought step by step:
1. First, identify ALL entities mentioned or implied in the text. Include:
   - People (full names)
   - Organizations, teams, companies
   - Locations, countries, nationalities
   - Dates (convert to YYYY-MM-DD format, e.g., "22 May 1980" → "1980-05-22")
   - Occupations, professions, roles (e.g., "volleyball player")
   - Any other significant concepts

2. Then, identify ALL relationships between these entities. For each relationship:
   - The subject and object MUST be different entities (no self-referential triples)
   - Use clear, descriptive relation names in lowercase_with_underscores format
   - Good: has_nationality, has_date_of_birth, has_occupation, member_of, plays_for
   - Bad: PERSON_BORN_ON, PERSON_NATIONALITY (don't embed entity types in relation names)
   - Distinguish between semantically different relationships
     (e.g., "member_of" for national team vs "plays_for" for club)

3. Also identify any INFERRED relationships that can be derived from the explicit ones.
   For example, if someone is part of "Greece men's national volleyball team", then
   that team has_nationality Greek.

Return JSON:
{{
    "reasoning": "<your step-by-step reasoning>",
    "entities": [
        {{
            "text": "<entity text>",
            "type": "<ENTITY_TYPE in UPPER_CASE>",
            "id": "<lowercase_underscored_id>"
        }}
    ],
    "triples": [
        {{
            "subject": "<subject entity text>",
            "relation": "<relation_name>",
            "object": "<object entity text>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text or 'inferred'>"
        }}
    ]
}}"""

        print("\n[1/3] Single-pass extraction (CoT)")
        print("-" * 50)

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert knowledge graph extractor. "
                        "Extract entities and relationships from text with high precision. "
                        "Always normalize dates to ISO 8601 (YYYY-MM-DD). "
                        "Never create self-referential triples where subject equals object. "
                        "Include inferred relationships when they follow logically."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model_tiers.get(ModelTier.MEDIUM, self.llm_config.model),
            temperature=0.2,
            max_tokens=4096,
        )

        entities = result.get("entities", [])
        triples = result.get("triples", [])

        # Filter self-referential triples
        triples = [
            t for t in triples
            if t.get("subject", "").strip().lower() != t.get("object", "").strip().lower()
        ]

        # Set default confidence
        for e in entities:
            e.setdefault("confidence", 0.8)
        for t in triples:
            t.setdefault("confidence", 0.8)

        context.entities = entities
        context.relations = triples
        results["entities_extracted"] = len(entities)
        results["triples_extracted"] = len(triples)
        results["domain"] = "auto"

        print(f"  Entities: {len(entities)}")
        print(f"  Triples: {len(triples)}")
        if result.get("reasoning"):
            print(f"  Reasoning: {result['reasoning'][:200]}...")

        # Step 2: Lightweight verification — single call
        print("\n[2/3] Lightweight verification")
        print("-" * 50)

        import json as _json
        triples_json_str = _json.dumps(triples, indent=2)
        verify_prompt = f"""Verify these knowledge graph triples against the source text.

SOURCE TEXT:
{text}

TRIPLES:
{triples_json_str}

For each triple:
- Mark as "verified" if supported by the text (explicit or reasonably inferred)
- Mark as "rejected" only if clearly wrong or contradicted
- Adjust confidence if needed

Return JSON:
{{
    "verified_triples": [
        {{
            "subject": "<subject>",
            "relation": "<relation>",
            "object": "<object>",
            "verified": true/false,
            "confidence": <0.0-1.0>
        }}
    ]
}}"""

        verify_result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fact verifier. Accept both explicit and reasonably inferred relationships.",
                },
                {"role": "user", "content": verify_prompt},
            ],
            model=self.model_tiers.get(ModelTier.MEDIUM, self.llm_config.model),
            temperature=0.1,
            max_tokens=4096,
        )

        verified = [
            t for t in verify_result.get("verified_triples", [])
            if t.get("verified", True)
        ]
        rejected_count = len(verify_result.get("verified_triples", [])) - len(verified)

        results["approved_triples"] = len(verified)
        results["rejected_triples"] = rejected_count
        print(f"  Approved: {len(verified)}, Rejected: {rejected_count}")

        # Step 3: Integrate into KG
        print("\n[3/3] Knowledge Graph Integration")
        print("-" * 50)

        integration_result = self.knowledge_organizer.run(
            context,
            entities=entities,
            triples=verified,
        )

        kg_stats = integration_result.metadata.get("kg_stats", {})
        results["kg_entities"] = kg_stats.get("total_entities", 0)
        results["kg_triples"] = kg_stats.get("total_triples", 0)

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        results["processing_time_seconds"] = elapsed
        results["voting_sessions"] = 0
        results["debates_triggered"] = 0
        results["items_accepted_by_vote"] = 0
        results["items_rejected_by_vote"] = 0
        results["refinement_iterations"] = 0

        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE (Single-Pass Mode)")
        print(f"{'='*70}")
        print(f"Document: {document_id}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Entities: {results['entities_extracted']} extracted -> {results['kg_entities']} in KG")
        print(f"Triples: {results['triples_extracted']} extracted -> {results['approved_triples']} approved -> {results['kg_triples']} in KG")
        print(f"{'='*70}\n")

        self.processing_history.append({
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })

        return results

    def _run_deliberation_phase(
        self,
        context: Any,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run multi-agent deliberation on extracted entities and triples.
        
        This is where the real multi-agent debate happens:
        1. Identify low-confidence items that need voting
        2. Submit hypotheses to the deliberation coordinator
        3. Collect votes from relevant agents
        4. Resolve debates for conflicting hypotheses
        5. Refine items based on deliberation outcomes
        
        Args:
            context: Processing context
            entities: Extracted entities
            triples: Linked triples
            segments: Document segments
            
        Returns:
            Deliberation results with refined entities/triples
        """
        from multi_agent_kg.core.deliberation import VoteType, DeliberationStatus
        
        results = {
            "voting_sessions": 0,
            "debates_triggered": 0,
            "accepted": 0,
            "rejected": 0,
            "refined_entities": None,
            "refined_triples": None,
        }
        
        # Track items to keep
        accepted_entities = []
        accepted_triples = []
        entity_hypothesis_ids = []
        triple_hypothesis_ids = []
        
        document_id = getattr(context, "document_id", None)
        
        # === ENTITY DELIBERATION ===
        # Only entities in the genuinely uncertain band (0.35–0.65) need voting.
        # Entities ≥ 0.65 are accepted directly; entities < 0.35 are rejected outright.
        # This prevents hundreds of pre-baked identical votes for items that are
        # clearly acceptable (confidence ~0.70) from flooding the log.
        ENTITY_UNCERTAIN_LOW  = 0.35
        ENTITY_UNCERTAIN_HIGH = 0.65
        low_confidence_entities = [
            e for e in entities
            if ENTITY_UNCERTAIN_LOW <= e.get("confidence", 1.0) < ENTITY_UNCERTAIN_HIGH
        ]
        # Items clearly below the floor → reject immediately, no vote needed
        for e in entities:
            if e.get("confidence", 1.0) < ENTITY_UNCERTAIN_LOW:
                results["rejected"] += 1
        # Items clearly above the ceiling → accept immediately
        high_confidence_entities = [
            e for e in entities
            if e.get("confidence", 1.0) >= ENTITY_UNCERTAIN_HIGH
        ]
        accepted_entities.extend(high_confidence_entities)
        
        if low_confidence_entities:
            print(f"  Deliberating on {len(low_confidence_entities)} low-confidence entities...")
            
            for entity in low_confidence_entities:
                # Submit hypothesis to deliberation coordinator
                hyp_id = self.deliberation_coordinator.submit_hypothesis(
                    author="EntityExtractor",
                    hypothesis_type="entity",
                    content={
                        "name": entity.get("name"),
                        "type": entity.get("type"),
                        "original_confidence": entity.get("confidence", 0.5),
                        "entity_data": entity,
                    },
                    confidence=entity.get("confidence", 0.5),
                    evidence=entity.get("evidence", []),
                    document_id=document_id,
                )
                entity_hypothesis_ids.append((hyp_id, entity))
                results["voting_sessions"] += 1
                
                # Simulate votes from other agents
                self._collect_entity_votes(hyp_id, entity, context)
        
        # === TRIPLE DELIBERATION ===
        TRIPLE_UNCERTAIN_LOW  = 0.35
        TRIPLE_UNCERTAIN_HIGH = 0.65
        low_confidence_triples = [
            t for t in triples
            if TRIPLE_UNCERTAIN_LOW <= t.get("confidence", 1.0) < TRIPLE_UNCERTAIN_HIGH
        ]
        for t in triples:
            if t.get("confidence", 1.0) < TRIPLE_UNCERTAIN_LOW:
                results["rejected"] += 1
        high_confidence_triples_direct = [
            t for t in triples
            if t.get("confidence", 1.0) >= TRIPLE_UNCERTAIN_HIGH
        ]
        accepted_triples.extend(high_confidence_triples_direct)
        
        if low_confidence_triples:
            print(f"  Deliberating on {len(low_confidence_triples)} low-confidence triples...")
            
            for triple in low_confidence_triples:
                subj = triple.get("subject", {}).get("name", "?")
                pred = triple.get("predicate", "?")
                obj = triple.get("object", {}).get("name", "?")
                
                hyp_id = self.deliberation_coordinator.submit_hypothesis(
                    author="RelationExtractor",
                    hypothesis_type="triple",
                    content={
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "original_confidence": triple.get("confidence", 0.5),
                        "triple_data": triple,
                    },
                    confidence=triple.get("confidence", 0.5),
                    evidence=triple.get("evidence", []),
                    document_id=document_id,
                )
                triple_hypothesis_ids.append((hyp_id, triple))
                results["voting_sessions"] += 1
                
                # Simulate votes from other agents
                self._collect_triple_votes(hyp_id, triple, context)
        
        # Process any remaining pending hypotheses
        self.deliberation_coordinator.process_pending(max_wait_seconds=0.1)
        
        # Collect results for entities
        for hyp_id, entity in entity_hypothesis_ids:
            hypothesis = self.deliberation_coordinator.hypotheses.get(hyp_id)
            if hypothesis:
                if hypothesis.status == DeliberationStatus.DEBATING:
                    # Resolve the debate
                    self._run_hypothesis_debate(hyp_id)
                    result = self.deliberation_coordinator.resolve_debate(hyp_id)
                    results["debates_triggered"] += 1
                    if result.get("accepted"):
                        results["accepted"] += 1
                        accepted_entities.append(entity)
                    else:
                        results["rejected"] += 1
                elif hypothesis.status == DeliberationStatus.ACCEPTED:
                    results["accepted"] += 1
                    accepted_entities.append(entity)
                elif hypothesis.status == DeliberationStatus.REJECTED:
                    results["rejected"] += 1
                else:
                    # Still pending, force resolution
                    self.deliberation_coordinator.force_resolution(hyp_id)
                    hypothesis = self.deliberation_coordinator.hypotheses.get(hyp_id)
                    if hypothesis and hypothesis.status == DeliberationStatus.ACCEPTED:
                        results["accepted"] += 1
                        accepted_entities.append(entity)
                    else:
                        results["rejected"] += 1
        
        # Collect results for triples
        for hyp_id, triple in triple_hypothesis_ids:
            hypothesis = self.deliberation_coordinator.hypotheses.get(hyp_id)
            if hypothesis:
                if hypothesis.status == DeliberationStatus.DEBATING:
                    self._run_hypothesis_debate(hyp_id)
                    result = self.deliberation_coordinator.resolve_debate(hyp_id)
                    results["debates_triggered"] += 1
                    if result.get("accepted"):
                        results["accepted"] += 1
                        accepted_triples.append(triple)
                    else:
                        results["rejected"] += 1
                elif hypothesis.status == DeliberationStatus.ACCEPTED:
                    results["accepted"] += 1
                    accepted_triples.append(triple)
                elif hypothesis.status == DeliberationStatus.REJECTED:
                    results["rejected"] += 1
                else:
                    self.deliberation_coordinator.force_resolution(hyp_id)
                    hypothesis = self.deliberation_coordinator.hypotheses.get(hyp_id)
                    if hypothesis and hypothesis.status == DeliberationStatus.ACCEPTED:
                        results["accepted"] += 1
                        accepted_triples.append(triple)
                    else:
                        results["rejected"] += 1
        
        # High-confidence items were already added at the start of this method
        # (band-filtering above). No duplicates needed here.
        
        # Return refined lists
        if low_confidence_entities:
            results["refined_entities"] = accepted_entities
        if low_confidence_triples:
            results["refined_triples"] = accepted_triples
        
        return results

    def _collect_entity_votes(
        self,
        hypothesis_id: str,
        entity: Dict[str, Any],
        context: Any,
    ) -> None:
        """Collect votes from agents on an entity hypothesis."""
        from multi_agent_kg.core.deliberation import VoteType
        
        confidence = entity.get("confidence", 0.5)
        has_evidence = bool(entity.get("evidence") or entity.get("source_spans"))
        entity_type = entity.get("type", "").lower()
        
        # Vote from DomainClassifier
        domain = getattr(context, "domain", "general")
        if domain == "scientific" and entity_type in ["protein", "gene", "chemical", "organism"]:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="DomainClassifier",
                vote_type=VoteType.ACCEPT,
                confidence=0.8,
                rationale=f"Entity type '{entity_type}' fits scientific domain",
            )
        elif confidence > 0.6:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="DomainClassifier",
                vote_type=VoteType.WEAK_ACCEPT,
                confidence=0.6,
                rationale=f"Confidence {confidence:.2f} acceptable",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="DomainClassifier",
                vote_type=VoteType.ABSTAIN,
                confidence=0.5,
                rationale=f"Low confidence {confidence:.2f}, uncertain",
            )
        
        # Vote from EvidenceLinker
        if has_evidence:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EvidenceLinker",
                vote_type=VoteType.ACCEPT,
                confidence=0.85,
                rationale="Entity has supporting evidence",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EvidenceLinker",
                vote_type=VoteType.WEAK_REJECT,
                confidence=0.6,
                rationale="No evidence linked to entity",
            )
        
        # Vote from RelationExtractor
        if confidence >= 0.5:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="RelationExtractor",
                vote_type=VoteType.WEAK_ACCEPT,
                confidence=0.7,
                rationale=f"Entity may participate in valid relations",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="RelationExtractor",
                vote_type=VoteType.ABSTAIN,
                confidence=0.4,
                rationale="Cannot assess entity for relation extraction",
            )

    def _collect_triple_votes(
        self,
        hypothesis_id: str,
        triple: Dict[str, Any],
        context: Any,
    ) -> None:
        """Collect votes from agents on a triple hypothesis."""
        from multi_agent_kg.core.deliberation import VoteType
        
        confidence = triple.get("confidence", 0.5)
        has_evidence = bool(triple.get("evidence") or triple.get("source_spans"))
        subj_conf = triple.get("subject", {}).get("confidence", 0.5)
        obj_conf = triple.get("object", {}).get("confidence", 0.5)
        
        # Vote from EntityExtractor
        if subj_conf > 0.6 and obj_conf > 0.6:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EntityExtractor",
                vote_type=VoteType.ACCEPT,
                confidence=0.8,
                rationale="Both subject and object are valid entities",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EntityExtractor",
                vote_type=VoteType.WEAK_REJECT,
                confidence=0.6,
                rationale=f"Subject ({subj_conf:.2f}) or object ({obj_conf:.2f}) has low confidence",
            )
        
        # Vote from EvidenceLinker
        if has_evidence:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EvidenceLinker",
                vote_type=VoteType.ACCEPT,
                confidence=0.9,
                rationale="Triple has strong supporting evidence",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="EvidenceLinker",
                vote_type=VoteType.REJECT,
                confidence=0.75,
                rationale="No evidence supports this triple",
            )
        
        # Vote from ExtractionValidator
        if confidence >= 0.55 and has_evidence:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="ExtractionValidator",
                vote_type=VoteType.ACCEPT,
                confidence=0.85,
                rationale="Triple meets quality standards",
            )
        elif confidence < 0.4:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="ExtractionValidator",
                vote_type=VoteType.REJECT,
                confidence=0.8,
                rationale=f"Triple confidence {confidence:.2f} too low",
            )
        else:
            self.deliberation_coordinator.receive_vote(
                hypothesis_id=hypothesis_id,
                voter="ExtractionValidator",
                vote_type=VoteType.ABSTAIN,
                confidence=0.5,
                rationale="Borderline quality, defer to other agents",
            )

    def _run_hypothesis_debate(self, hypothesis_id: str) -> None:
        """Run a debate on a hypothesis by collecting arguments."""
        hypothesis = self.deliberation_coordinator.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return
        
        # Collect debate arguments from participating agents
        if hypothesis.hypothesis_type == "entity":
            # EntityExtractor supports
            self.deliberation_coordinator.receive_debate_argument(
                hypothesis_id=hypothesis_id,
                agent="EntityExtractor",
                position="support",
                argument=f"Entity was extracted with initial confidence {hypothesis.initial_confidence:.2f}",
            )
            # EvidenceLinker based on evidence
            if hypothesis.evidence:
                self.deliberation_coordinator.receive_debate_argument(
                    hypothesis_id=hypothesis_id,
                    agent="EvidenceLinker",
                    position="support",
                    argument=f"Entity has {len(hypothesis.evidence)} supporting evidence spans",
                )
            else:
                self.deliberation_coordinator.receive_debate_argument(
                    hypothesis_id=hypothesis_id,
                    agent="EvidenceLinker",
                    position="oppose",
                    argument="Entity lacks supporting evidence in the document",
                )
        else:  # triple
            # RelationExtractor supports
            self.deliberation_coordinator.receive_debate_argument(
                hypothesis_id=hypothesis_id,
                agent="RelationExtractor",
                position="support",
                argument=f"Relation extracted with confidence {hypothesis.initial_confidence:.2f}",
            )
            # EvidenceLinker based on evidence
            if hypothesis.evidence:
                self.deliberation_coordinator.receive_debate_argument(
                    hypothesis_id=hypothesis_id,
                    agent="EvidenceLinker",
                    position="support",
                    argument=f"Relation has {len(hypothesis.evidence)} supporting evidence spans",
                )
            else:
                self.deliberation_coordinator.receive_debate_argument(
                    hypothesis_id=hypothesis_id,
                    agent="EvidenceLinker",
                    position="oppose",
                    argument="Relation lacks textual evidence",
                )

    def process_corpus(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple documents as a corpus.
        
        Args:
            documents: List of dicts with 'text' and optional 'id', 'metadata'
            
        Returns:
            Aggregate results
        """
        print("\n" + "=" * 70)
        print(f"PROCESSING CORPUS: {len(documents)} documents")
        print("=" * 70)
        
        all_results = []
        
        for i, doc in enumerate(documents):
            print(f"\n[Document {i+1}/{len(documents)}]")
            result = self.process_document(
                text=doc.get("text"),
                source_path=doc.get("source"),
                document_id=doc.get("id"),
                metadata=doc.get("metadata"),
            )
            all_results.append(result)
        
        # Cross-document entity resolution
        if self.enable_cross_document:
            print("\n" + "-" * 50)
            print("Cross-Document Entity Resolution")
            print("-" * 50)
            self._resolve_cross_document_entities()
        
        # Aggregate stats
        aggregate = {
            "documents_processed": len(documents),
            "total_entities": sum(r.get("kg_entities", 0) for r in all_results),
            "total_triples": sum(r.get("kg_triples", 0) for r in all_results),
            "total_time": sum(r.get("processing_time_seconds", 0) for r in all_results),
            "memory_stats": self.shared_memory.get_stats(),
            "kg_stats": self.knowledge_organizer.get_kg_stats(),
        }
        
        print("\n" + "=" * 70)
        print("CORPUS PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Documents: {aggregate['documents_processed']}")
        print(f"Total Entities: {aggregate['total_entities']}")
        print(f"Total Triples: {aggregate['total_triples']}")
        print(f"Total Time: {aggregate['total_time']:.2f}s")
        print("=" * 70 + "\n")
        
        return aggregate

    def _resolve_cross_document_entities(self) -> None:
        """Resolve entities across documents."""
        # This uses the shared memory's entity alias system
        stats = self.shared_memory.get_stats()
        print(f"  Entity aliases registered: {stats.get('entity_aliases', 0)}")
        print(f"  Unique entities tracked: {stats.get('unique_entities', 0)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "session_start": self.session_start.isoformat(),
            "documents_processed": self.document_count,
            "memory_stats": self.shared_memory.get_stats(),
            "kg_stats": self.knowledge_organizer.get_kg_stats(),
            "agent_stats": {
                "document_processor": self.document_processor.get_stats(),
                "domain_classifier": self.domain_classifier.get_stats(),
                "entity_extractor": self.entity_extractor.get_stats(),
                "relation_extractor": self.relation_extractor.get_stats(),
                "evidence_linker": self.evidence_linker.get_stats(),
                "extraction_validator": self.extraction_validator.get_stats(),
                "verification_agent": self.verification_agent.get_stats(),
                "knowledge_organizer": self.knowledge_organizer.get_stats(),
            },
            "discovered_relations": list(self.relation_extractor.get_discovered_relations().keys()),
        }

    def visualize_kg(
        self,
        output_file: str = "kg_visualization.html",
        layout: str = "spring",
        show_labels: bool = True,
        generate_static: bool = False
    ) -> str:
        """
        Visualize the knowledge graph.
        
        Args:
            output_file: Output file path (default: kg_visualization.html)
            layout: Layout algorithm - spring, hierarchical, circular, kamada_kawai
            show_labels: Whether to show edge labels
            generate_static: Also generate static PNG version
            
        Returns:
            Path to generated visualization file
            
        Raises:
            ImportError: If visualization dependencies not installed
        """
        if not VISUALIZER_AVAILABLE:
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install pyvis networkx matplotlib"
            )
        
        kg = self.knowledge_organizer.knowledge_graph
        visualizer = KGVisualizer(kg)
        
        print(f"\n📊 Generating knowledge graph visualization...")
        print(f"  Layout: {layout}")
        print(f"  Output: {output_file}")
        
        # Generate interactive HTML
        visualizer.visualize_kg(
            output_file=output_file,
            layout=layout,
            show_labels=show_labels
        )
        
        # Generate static PNG if requested
        if generate_static:
            static_file = output_file.replace(".html", ".png")
            print(f"  Static: {static_file}")
            visualizer.visualize_kg(
                output_file=static_file,
                layout=layout,
                show_labels=show_labels
            )
        
        print("✓ Visualization complete!")
        return output_file

    def export(self) -> Dict[str, Any]:
        """Export complete system state."""
        return {
            "knowledge_graph": self.knowledge_organizer.export_knowledge_graph(),
            "memory": self.shared_memory.export(),
            "stats": self.get_stats(),
            "processing_history": self.processing_history,
        }
