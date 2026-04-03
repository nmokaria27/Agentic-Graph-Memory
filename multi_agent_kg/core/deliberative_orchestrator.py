"""
Deliberative Multi-Agent Orchestrator.

Research-driven pipeline integrating:
- Adaptive planning (Phase 8) — selects strategy per document
- Constrained decoding via Pydantic schemas (eliminates JSON failures)
- GLiNER/GLiREL zero-cost first pass (Phase 2)
- Consolidated extraction + GraphRAG-style gleaning (Phase 3)
- KGGen entity resolution clustering (Phase 4)
- FinReflectKG critic-corrector verification loop (Phase 5)
- Triplex parallel extraction + schema alignment (Phase 6)

Architecture (11 stages):
  [1]  AdaptivePlanner          -> strategy selection (batch size, gleaning, etc.)
  [2]  DocumentProcessor        -> segments (10% overlap)
  [3]  DomainClassifier         -> domain + schema (<=7 entity types, <=15 relations)
  [4]  FastExtractor (optional) -> baseline entities + triples (GLiNER, no LLM)
  [5]  EntityExtractor          -> refined entities (consolidated + gleaning)
  [6]  EntityResolver           -> deduplicated entities (KGGen clustering)
  [7]  RelationExtractor        -> triples (consolidated + gleaning)
  [8]  Triplex + SchemaAligner  -> merged parallel extractions (optional)
  [9]  CriticCorrectorLoop      -> verified entities + triples (1-2 iterations)
  [10] OrphanLinker             -> rescue disconnected entities (link/reify/prune)
  [11] KnowledgeOrganizer       -> final KG
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import os

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CollaborationProtocol
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.deliberation import DeliberationCoordinator, VoteType

from multi_agent_kg.agents.base import AgentContext, ModelTier, AGENT_MODEL_OVERRIDES
from multi_agent_kg.agents.document_processor import DocumentProcessor
from multi_agent_kg.agents.domain_classifier import DomainClassifier
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.agents.evidence_linker import EvidenceLinker
from multi_agent_kg.agents.extraction_validator import ExtractionValidator
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer
from multi_agent_kg.agents.entity_resolver import EntityResolver
from multi_agent_kg.agents.critic_agent import CriticAgent
from multi_agent_kg.agents.corrector_agent import CorrectorAgent
from multi_agent_kg.agents.orphan_linker import OrphanLinker
from multi_agent_kg.core.relation_library import RelationLibrary
from multi_agent_kg.core.adaptive_planner import AdaptivePlanner
from multi_agent_kg.utils.progress import PipelineProgress

# Optional: GLiNER-based fast extractor
try:
    from multi_agent_kg.agents.fast_extractor import FastExtractor
    FAST_EXTRACTOR_AVAILABLE = True
except ImportError:
    FAST_EXTRACTOR_AVAILABLE = False

# Optional: Triplex parallel extractor + Schema Aligner (Phase 6)
try:
    from multi_agent_kg.agents.triplex_extractor import TriplexExtractor
    TRIPLEX_AVAILABLE = True
except ImportError:
    TRIPLEX_AVAILABLE = False

try:
    from multi_agent_kg.agents.schema_aligner import SchemaAligner
    SCHEMA_ALIGNER_AVAILABLE = True
except ImportError:
    SCHEMA_ALIGNER_AVAILABLE = False

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
        max_refinement_iterations: int = 2,
        enable_self_consistency: bool = True,
        enable_open_world: bool = True,
        enable_cross_document: bool = True,
        enable_deliberation: bool = True,
        enable_fast_first_pass: bool = True,
        enable_critic_corrector: bool = True,
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
        self.enable_fast_first_pass = enable_fast_first_pass
        self.enable_critic_corrector = enable_critic_corrector
        self.enable_triplex = TRIPLEX_AVAILABLE
        self.debug_logger = debug_logger

        # Adaptive planner — selects strategy per document
        self.adaptive_planner = AdaptivePlanner()

        # Relation Library: persistent cross-document relation catalog
        self.relation_library = RelationLibrary()
        relation_library_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "relation_library.json",
        )
        self._relation_library_path = os.path.normpath(relation_library_path)
        if os.path.exists(self._relation_library_path):
            self.relation_library = RelationLibrary.load(self._relation_library_path)
            print(f"  Loaded Relation Library: {self.relation_library.size} known relation types")
        
        # Model tier configuration
        self.model_tiers = model_tiers or {
            ModelTier.SMALL: "gemma3:27b",
            ModelTier.MEDIUM: "gemma3:27b",
            ModelTier.LARGE: "gemma3:27b",
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

        # Rich progress display
        self.progress = PipelineProgress(total_stages=11)
        self._print_header()

    def _init_agents(self) -> None:
        """Initialize all agents with shared infrastructure."""
        
        # Worker Agents
        self.document_processor = DocumentProcessor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        
        self.domain_classifier = DomainClassifier(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        
        self.entity_extractor = EntityExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
        )
        
        self.relation_extractor = RelationExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
            enable_open_world=self.enable_open_world,
        )
        
        self.evidence_linker = EvidenceLinker(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            enable_cross_reference=self.enable_cross_document,
        )
        
        # Coordinator Agents
        self.extraction_validator = ExtractionValidator(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            max_iterations=self.max_refinement_iterations,
        )
        
        self.verification_agent = ExtractionVerificationAgent(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
        )
        
        self.knowledge_organizer = KnowledgeOrganizer(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        
        # New agents (Phases 2, 4, 5)
        self.fast_extractor = None
        if self.enable_fast_first_pass and FAST_EXTRACTOR_AVAILABLE:
            self.fast_extractor = FastExtractor(
                knowledge_graph=self.knowledge_graph,
                shared_memory=self.shared_memory,
                message_bus=self.message_bus,
                llm_config=self.llm_config,
            )

        self.entity_resolver = EntityResolver(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )

        self.critic_agent = CriticAgent(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )

        self.corrector_agent = CorrectorAgent(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )

        # Orphan Linker: rescues disconnected entities post-verification
        self.orphan_linker = OrphanLinker(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )

        # Phase 6: Triplex + SchemaAligner (optional)
        self.triplex_extractor = None
        if self.enable_triplex and TRIPLEX_AVAILABLE:
            self.triplex_extractor = TriplexExtractor(
                knowledge_graph=self.knowledge_graph,
                shared_memory=self.shared_memory,
                message_bus=self.message_bus,
                llm_config=self.llm_config,
            )

        self.schema_aligner = None
        if SCHEMA_ALIGNER_AVAILABLE:
            self.schema_aligner = SchemaAligner(
                knowledge_graph=self.knowledge_graph,
                shared_memory=self.shared_memory,
                message_bus=self.message_bus,
                llm_config=self.llm_config,
            )

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
        """Print orchestrator header using Rich progress display."""
        self.progress.print_header({
            "model_tiers": {t.value: m for t, m in self.model_tiers.items()},
            "agent_models": {k: v for k, v in sorted(AGENT_MODEL_OVERRIDES.items()) if v},
            "features": {
                "Self-Consistency": self.enable_self_consistency,
                "Open-World Relations": self.enable_open_world,
                "Cross-Document Resolution": self.enable_cross_document,
                "Multi-Agent Deliberation": self.enable_deliberation,
                "Fast First Pass (GLiNER)": self.enable_fast_first_pass,
                "Triplex Parallel Extraction": self.enable_triplex,
                "Critic-Corrector Loop": self.enable_critic_corrector,
                "Adaptive Planning": True,
            },
            "quality": {
                "Threshold": self.quality_threshold,
                "Max Iterations": self.max_refinement_iterations,
            },
        })

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
        
        self.progress.print_document_header(document_id, self.document_count, self.document_count)
        
        # Create context
        context = AgentContext(
            document_id=document_id,
            text=text or "",
            quality_threshold=self.quality_threshold,
            max_iterations=self.max_refinement_iterations,
        )
        
        results = {}

        p = self.progress

        # ===== STEP 1: Adaptive Planning =====
        p.stage_start(1, "Adaptive Planning")
        strategy = self.adaptive_planner.analyze(text or "", num_documents=1)
        results["strategy"] = strategy.name
        p.stage_complete(1, {"Strategy": strategy.name, "Reason": strategy.reasoning[:60]})

        # Apply strategy overrides
        effective_gleanings = strategy.max_gleanings
        effective_critic_iters = strategy.max_critic_iterations

        # ===== STEP 2: Document Processing =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(2, "Document Processing")
        p.stage_start(2, "Document Processing")
        doc_result = self.document_processor.run(context, source_path=source_path)
        segments = doc_result.items
        results["segments"] = len(segments)
        p.stage_complete(2, {"Segments": len(segments)})

        # ===== STEP 3: Domain Classification =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(3, "Domain Classification")
        p.stage_start(3, "Domain Classification")
        domain_result = self.domain_classifier.run(context, segments=segments)
        domain_config = domain_result.items[0] if domain_result.items else {}
        context.domain = domain_config.get("primary_domain", domain_config.get("domain", "general"))
        results["domain"] = context.domain
        entity_type_count = len(domain_config.get("entity_types", []))
        relation_type_count = len(domain_config.get("relation_types", []))
        p.stage_complete(3, {
            "Domain": context.domain,
            "Schema": f"{entity_type_count} entity types, {relation_type_count} relations",
        })

        # ===== STEP 4: Fast First Pass (GLiNER/GLiREL, optional) =====
        prior_entities = None
        prior_triples = None
        if self.fast_extractor:
            if self.debug_logger:
                self.debug_logger.log_stage_header(4, "Fast First Pass (GLiNER/GLiREL)")
            p.stage_start(4, "Fast First Pass", "GLiNER/GLiREL zero-cost extraction")
            fast_result = self.fast_extractor.run(
                context, segments=segments, domain_config=domain_config,
            )
            prior_entities = fast_result.items if fast_result.items else None
            prior_triples = fast_result.metadata.get("triples", None)
            results["fast_entities"] = len(prior_entities or [])
            results["fast_triples"] = len(prior_triples or [])
            p.stage_complete(4, {
                "Entities": len(prior_entities or []),
                "Triples": len(prior_triples or []),
            })
        else:
            p.stage_start(4, "Fast First Pass", "skipped (GLiNER not installed)")
            results["fast_entities"] = 0
            results["fast_triples"] = 0
            p.stage_complete(4, {"Status": "skipped"})

        # ===== STEP 5: Entity Extraction (consolidated + gleaning) =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(5, "Entity Extraction")
        p.stage_start(5, "Entity Extraction", "consolidated + gleaning")
        entity_result = self.entity_extractor.run(
            context,
            segments=segments,
            domain_config=domain_config,
            prior_entities=prior_entities,
        )
        entities = entity_result.items
        context.entities = entities
        results["entities_extracted"] = len(entities)
        p.stage_complete(5, {
            "Entities": len(entities),
            "Confidence": f"{entity_result.confidence:.2f}",
        })

        # ===== STEP 6: Entity Resolution (KGGen clustering) =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(6, "Entity Resolution")
        p.stage_start(6, "Entity Resolution", "KGGen clustering + canonicalization")
        resolve_result = self.entity_resolver.run(
            context,
            entities=entities,
            domain_config=domain_config,
        )
        resolved_entities = resolve_result.items
        context.entities = resolved_entities
        results["entities_resolved"] = len(resolved_entities)
        results["entities_merged"] = len(entities) - len(resolved_entities)
        p.stage_complete(6, {
            "Resolved": len(resolved_entities),
            "Merged": results["entities_merged"],
        })

        # ===== STEP 7: Relation Extraction (consolidated + gleaning) =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(7, "Relation Extraction")
        p.stage_start(7, "Relation Extraction", "consolidated + gleaning")
        relation_result = self.relation_extractor.run(
            context,
            segments=segments,
            entities=resolved_entities,
            domain_config=domain_config,
            prior_triples=prior_triples,
        )
        triples = relation_result.items
        context.relations = triples
        results["triples_extracted"] = len(triples)

        # Populate Relation Library
        for triple in triples:
            rel_name = triple.get("relation", "")
            if rel_name:
                self.relation_library.register(
                    name=rel_name,
                    document_id=document_id,
                    example={
                        "subject": triple.get("subject", ""),
                        "object": triple.get("object", ""),
                        "evidence": triple.get("evidence", ""),
                    },
                )
        # Merge similar relations in the library
        self.relation_library.merge_similar()
        p.stage_complete(7, {
            "Triples": len(triples),
            "Relation types": self.relation_library.size,
        })

        # ===== STEP 8: Triplex + Schema Alignment (optional, Phase 6) =====
        triplex_entities = []
        triplex_triples = []
        if self.triplex_extractor and strategy.enable_triplex:
            if self.debug_logger:
                self.debug_logger.log_stage_header(8, "Triplex + Schema Alignment")
            p.stage_start(8, "Triplex + Schema Alignment", "parallel extraction merge")
            triplex_result = self.triplex_extractor.run(
                context, segments=segments, domain_config=domain_config,
            )
            # Separate entities and triples from triplex items
            for item in (triplex_result.items or []):
                if item.get("item_type") == "entity":
                    triplex_entities.append(item)
                elif item.get("item_type") == "triple":
                    triplex_triples.append(item)
            results["triplex_entities"] = len(triplex_entities)
            results["triplex_triples"] = len(triplex_triples)

            # Merge via SchemaAligner if available
            if self.schema_aligner and (triplex_entities or triplex_triples):
                align_result = self.schema_aligner.run(
                    context,
                    main_entities=resolved_entities,
                    main_triples=triples,
                    triplex_entities=triplex_entities,
                    triplex_triples=triplex_triples,
                )
                merged = align_result.items
                if isinstance(merged, dict):
                    resolved_entities = merged.get("entities", resolved_entities)
                    triples = merged.get("triples", triples)
                    context.entities = resolved_entities
                    context.relations = triples
                results["aligned_entities"] = len(resolved_entities)
                results["aligned_triples"] = len(triples)
            p.stage_complete(8, {
                "Triplex entities": len(triplex_entities),
                "Triplex triples": len(triplex_triples),
            })
        else:
            p.stage_start(8, "Triplex + Schema Alignment", "skipped")
            results["triplex_entities"] = 0
            results["triplex_triples"] = 0
            p.stage_complete(8, {"Status": "skipped"})

        # ===== STEP 9: Critic-Corrector Verification Loop =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(9, "Critic-Corrector Verification")
        p.stage_start(9, "Verification", "critic-corrector reflection loop")

        verified_entities = resolved_entities
        verified_triples = triples
        results["critic_iterations"] = 0

        if self.enable_critic_corrector:
            verified_entities, verified_triples, iterations = self._run_critic_corrector_loop(
                context, resolved_entities, triples,
            )
            results["critic_iterations"] = iterations
            results["approved_triples"] = len(verified_triples)
            results["rejected_triples"] = len(triples) - len(verified_triples)
            p.stage_complete(9, {
                "Iterations": iterations,
                "Approved triples": len(verified_triples),
            })
        else:
            validation_result = self.extraction_validator.run(
                context, entities=resolved_entities, triples=triples,
            )
            validated = validation_result.items
            verification_result = self.verification_agent.run(
                context,
                entities=validated.get("entities", resolved_entities),
                triples=validated.get("triples", triples),
            )
            verified = verification_result.items
            verified_entities = verified.get("entities", resolved_entities)
            verified_triples = verified.get("approved_triples", triples)
            results["approved_triples"] = len(verified_triples)
            results["rejected_triples"] = len(verified.get("rejected_triples", []))
            p.stage_complete(9, {"Approved (legacy)": results["approved_triples"]})

        # ===== STEP 10: Orphan Linker =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(10, "Orphan Linking")
        p.stage_start(10, "Orphan Linking", "rescuing disconnected entities")

        try:
            orphan_result = self.orphan_linker.run(
                context,
                entities=verified_entities,
                triples=verified_triples,
            )
            orphan_items = orphan_result.items
            verified_entities = orphan_items.get("entities", verified_entities)
            verified_triples = orphan_items.get("triples", verified_triples)
            orphan_meta = orphan_result.metadata
            results["orphans_found"] = orphan_meta.get("orphans_found", 0)
            results["orphans_linked"] = orphan_meta.get("linked", 0)
            results["orphans_reified"] = orphan_meta.get("reified", 0)
            results["orphans_pruned"] = orphan_meta.get("pruned", 0)
            results["orphan_new_triples"] = orphan_meta.get("new_triples_added", 0)
            p.stage_complete(10, {
                "Orphans found": results["orphans_found"],
                "Linked": results["orphans_linked"],
                "Reified": results["orphans_reified"],
                "Pruned": results["orphans_pruned"],
                "New triples": results["orphan_new_triples"],
            })
        except Exception as exc:
            # Safe fallback: if orphan linker fails, proceed without it
            if self.debug_logger:
                self.debug_logger.log(f"OrphanLinker failed, proceeding without: {exc}")
            else:
                print(f"  [WARN] OrphanLinker failed, proceeding without: {exc}")
            results["orphans_found"] = 0
            p.stage_complete(10, {"Status": "skipped (error)"})

        # ===== STEP 11: Knowledge Organization =====
        if self.debug_logger:
            self.debug_logger.log_stage_header(11, "Knowledge Graph Integration")
        p.stage_start(11, "Knowledge Graph Integration")
        integration_result = self.knowledge_organizer.run(
            context,
            entities=verified_entities,
            triples=verified_triples,
        )
        kg_stats = integration_result.metadata.get("kg_stats", {})
        results["kg_entities"] = kg_stats.get("total_entities", 0)
        results["kg_triples"] = kg_stats.get("total_triples", 0)
        p.stage_complete(11, {
            "KG Entities": results["kg_entities"],
            "KG Triples": results["kg_triples"],
        })

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        results["processing_time_seconds"] = elapsed
        results["relation_library_size"] = self.relation_library.size
        p.print_summary(results)

        # Persist relation library
        try:
            self.relation_library.save(self._relation_library_path)
        except Exception:
            pass  # best-effort

        # Store in history
        self.processing_history.append({
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })

        return results

    def _run_critic_corrector_loop(
        self,
        context: AgentContext,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """
        Run the FinReflectKG-style critic-corrector verification loop.

        Iterates: critic reviews → corrector fixes → repeat until no issues or max iterations.

        Args:
            context: Processing context
            entities: Extracted entities
            triples: Extracted triples

        Returns:
            Tuple of (verified_entities, verified_triples, iterations_run)
        """
        current_entities = entities
        current_triples = triples
        max_iterations = self.max_refinement_iterations

        for iteration in range(max_iterations):
            self.progress.stage_detail(f"Iteration {iteration + 1}/{max_iterations}")

            # Critic reviews
            critic_result = self.critic_agent.run(
                context,
                entities=current_entities,
                triples=current_triples,
            )
            feedback = critic_result.items
            critic_quality = critic_result.confidence

            # Check if critic found issues
            issues = feedback.get("issues", []) if isinstance(feedback, dict) else []
            entities_ok = feedback.get("entities_ok", True) if isinstance(feedback, dict) else True
            triples_ok = feedback.get("triples_ok", True) if isinstance(feedback, dict) else True

            if (entities_ok and triples_ok) or not issues:
                self.progress.stage_detail(f"Critic: no issues (quality: {critic_quality:.2f})")
                return current_entities, current_triples, iteration + 1

            self.progress.stage_detail(f"Critic: {len(issues)} issues (quality: {critic_quality:.2f})")

            # Corrector fixes
            corrector_result = self.corrector_agent.run(
                context,
                entities=current_entities,
                triples=current_triples,
                critic_feedback=feedback,
            )
            corrected = corrector_result.items

            # Update with corrections
            if isinstance(corrected, dict):
                if corrected.get("corrected_entities"):
                    current_entities = corrected["corrected_entities"]
                if corrected.get("corrected_triples"):
                    current_triples = corrected["corrected_triples"]

            issues_addressed = corrected.get("issues_addressed", 0) if isinstance(corrected, dict) else 0
            self.progress.stage_detail(f"Corrector: fixed {issues_addressed} issues")

        return current_entities, current_triples, max_iterations

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
        all_results = []

        for i, doc in enumerate(documents):
            self.progress.print_document_header(
                doc.get("id", f"doc_{i+1}"), i + 1, len(documents),
            )
            result = self.process_document(
                text=doc.get("text"),
                source_path=doc.get("source"),
                document_id=doc.get("id"),
                metadata=doc.get("metadata"),
            )
            all_results.append(result)

        # Cross-document entity resolution
        if self.enable_cross_document:
            self._resolve_cross_document_entities()

        # Aggregate stats
        aggregate = {
            "Documents processed": len(documents),
            "Total entities": sum(r.get("kg_entities", 0) for r in all_results),
            "Total triples": sum(r.get("kg_triples", 0) for r in all_results),
            "Total time": f"{sum(r.get('processing_time_seconds', 0) for r in all_results):.1f}s",
            "Relation Library": f"{self.relation_library.size} types",
        }

        self.progress.print_corpus_summary(aggregate)

        return aggregate

    def _resolve_cross_document_entities(self) -> None:
        """Resolve entities across documents."""
        stats = self.shared_memory.get_stats()
        self.progress.stage_detail(
            f"Cross-doc resolution: {stats.get('entity_aliases', 0)} aliases, "
            f"{stats.get('unique_entities', 0)} unique entities"
        )

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
