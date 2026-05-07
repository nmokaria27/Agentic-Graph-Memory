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
import os

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph, GovernanceDecision
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CollaborationProtocol
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.deliberation import DeliberationCoordinator, VoteType
from multi_agent_kg.core.domain_builder import DomainBuilder

from multi_agent_kg.agents.base import AgentContext, ModelTier
from multi_agent_kg.agents.document_processor import DocumentProcessor
from multi_agent_kg.agents.domain_classifier import DomainClassifier
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.relation_extractor import RelationExtractor, SCIERC_RELATION_ALIASES
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
        governed_kg: Optional[GovernedKnowledgeGraph] = None,
        enable_governance: bool = True,
        governance_mode: str = "audit_only",
        reuse_corpus_schema: bool = False,
        expand_org_chart_with_schema: bool = False,
        continue_on_document_error: bool = True,
        skip_evidence_linking: bool = False,
        skip_verification: bool = False,
        strict_source_only_verification: bool = False,
        quality_threshold: float = 0.60,
        max_refinement_iterations: int = 4,
        enable_self_consistency: bool = True,
        enable_open_world: bool = True,
        enable_fixed_schema_pairwise: bool = True,
        enable_deterministic_value_harvesting: bool = False,
        enable_deterministic_attribute_binding: bool = False,
        enable_cross_document: bool = True,
        enable_deliberation: bool = True,
        model_tiers: Optional[Dict[ModelTier, str]] = None,
        target_num_domains: Optional[int] = None,
        debug_logger = None,
        schema_override: Optional[Dict[str, Any]] = None,
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
            enable_deterministic_value_harvesting: Opt-in rule-based
                open-world value candidate harvesting. Disabled by default
                to keep the general creation path schema/LLM-driven.
            enable_deterministic_attribute_binding: Opt-in rule-based
                value-to-subject relation binding. Disabled by default; the
                general extraction path should use discovered/LLM relations.
            expand_org_chart_with_schema: In open-world governed builds, add
                newly discovered schema domains to the existing org chart
                instead of freezing ownership after the first document.
            enable_fixed_schema_pairwise: In fixed-schema mode, add pairwise
                relation scoring over local entity pairs
            enable_cross_document: Enable cross-document entity resolution
            enable_deliberation: Enable multi-agent voting and debate
            model_tiers: Custom model tier mapping
            target_num_domains: Optional cap/target for schema-bootstrap
                domains. Useful for QA corpora where stable broad domain
                ownership is better than per-document micro-domains.
            debug_logger: Debug logger for tracking communications
            schema_override: If provided, skip dynamic schema discovery and
                use these fixed entity_types and relation_types. Format:
                {"entity_types": [{"type": "...", "description": "..."}],
                 "relation_types": [{"type": "...", "description": "..."}]}
        """
        self.llm_config = llm_config or LLMConfig()
        self.enable_governance = enable_governance or governed_kg is not None
        if governed_kg is not None:
            self.governed_kg = governed_kg
            self.knowledge_graph = governed_kg.kg
        elif self.enable_governance:
            self.knowledge_graph = knowledge_graph or KnowledgeGraph()
            self.governed_kg = GovernedKnowledgeGraph(
                kg=self.knowledge_graph,
                governance_mode=governance_mode,
            )
        else:
            self.knowledge_graph = knowledge_graph or KnowledgeGraph()
            self.governed_kg = None
        self.governance_mode = (
            self.governed_kg.governance_mode if self.governed_kg is not None else "disabled"
        )
        self.reuse_corpus_schema = reuse_corpus_schema
        self.expand_org_chart_with_schema = expand_org_chart_with_schema
        self.continue_on_document_error = continue_on_document_error
        self.skip_evidence_linking = skip_evidence_linking
        self.skip_verification = skip_verification
        self.strict_source_only_verification = strict_source_only_verification
        self.quality_threshold = quality_threshold
        self.max_refinement_iterations = max_refinement_iterations
        self.enable_self_consistency = enable_self_consistency
        self.enable_open_world = enable_open_world
        self.enable_fixed_schema_pairwise = enable_fixed_schema_pairwise
        self.enable_deterministic_value_harvesting = enable_deterministic_value_harvesting
        self.enable_deterministic_attribute_binding = enable_deterministic_attribute_binding
        self.enable_cross_document = enable_cross_document
        self.enable_deliberation = enable_deliberation
        self.debug_logger = debug_logger
        self.schema_override = schema_override
        self.target_num_domains = target_num_domains
        self.domain_builder = DomainBuilder(self.llm_config, target_num_domains=target_num_domains)
        self._active_source_text = ""
        self._strict_review_board = None
        
        # Model tier configuration
        self.model_tiers = model_tiers or {
            ModelTier.SMALL: os.getenv("LLM_SMALL_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
            ModelTier.MEDIUM: os.getenv("LLM_MEDIUM_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
            ModelTier.LARGE: os.getenv("LLM_LARGE_MODEL", os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")),
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
        self._corpus_domain_config: Optional[Dict[str, Any]] = None
        
        self._print_header()
        self._configure_governance_review()

    def _canonicalize_schema_relation(self, relation: Any) -> Optional[str]:
        if not self.schema_override:
            return str(relation).strip() if relation else None
        allowed = [
            item.get("type") if isinstance(item, dict) else str(item)
            for item in self.schema_override.get("relation_types", [])
        ]
        allowed = [label.strip() for label in allowed if label and str(label).strip()]
        if not allowed or relation is None:
            return str(relation).strip() if relation else None
        relation_text = str(relation).strip()
        norm = relation_text.upper().replace("-", "_").replace(" ", "_")
        allowed_by_norm = {
            label.upper().replace("-", "_").replace(" ", "_"): label
            for label in allowed
        }
        if norm in allowed_by_norm:
            return allowed_by_norm[norm]
        alias = SCIERC_RELATION_ALIASES.get(norm)
        if alias in allowed:
            return alias
        return None

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
            use_self_consistency=self.enable_self_consistency,
        )
        
        self.entity_extractor = EntityExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
            enable_deterministic_value_harvesting=self.enable_deterministic_value_harvesting,
        )
        
        self.relation_extractor = RelationExtractor(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            use_self_consistency=self.enable_self_consistency,
            enable_open_world=self.enable_open_world,
            enable_fixed_schema_pairwise=self.enable_fixed_schema_pairwise,
            enable_deterministic_attribute_binding=self.enable_deterministic_attribute_binding,
        )
        
        self.evidence_linker = EvidenceLinker(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
            quality_threshold=self.quality_threshold,
            enable_cross_reference=self.enable_cross_document,
            strict_source_only=self.strict_source_only_verification,
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
            strict_source_only=self.strict_source_only_verification,
        )
        
        self.knowledge_organizer = KnowledgeOrganizer(
            knowledge_graph=self.knowledge_graph,
            governed_kg=self.governed_kg,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        
        # Visualizer will be initialized on-demand when export() is called
        self.visualizer = None

        # Propagate orchestrator model_tiers to every agent so --model truly routes
        # to the LLM at runtime (agent subclasses don't forward model_tiers via super()).
        for agent in (
            self.document_processor,
            self.domain_classifier,
            self.entity_extractor,
            self.relation_extractor,
            self.evidence_linker,
            self.extraction_validator,
            self.verification_agent,
            self.knowledge_organizer,
        ):
            agent.model_tiers = self.model_tiers

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

    def _configure_governance_review(self) -> None:
        """Install LLM-backed strict review for extraction-time governance."""
        if self.governed_kg is None:
            self._strict_review_board = None
            return
        if self.governance_mode not in {"strict", "triage"}:
            self.governed_kg.set_review_callback(None)
            self._strict_review_board = None
            return

        from multi_agent_kg.core.incremental_enrichment import GovernanceReviewBoard

        self._strict_review_board = GovernanceReviewBoard(
            self.governed_kg.org_chart,
            self.knowledge_graph,
            self.llm_config,
        )

        def review_callback(
            triple: Triple,
            assignment: Any,
            kg: KnowledgeGraph,
            org_chart: Any,
        ) -> GovernanceDecision:
            # Keep one review board instance per extraction run so strict-mode
            # governance can maintain consistency across multiple decisions.
            self._strict_review_board.org_chart = org_chart
            self._strict_review_board.base_kg = kg
            result = self._strict_review_board._review_candidate(
                candidate=triple,
                assignment=assignment,
                source_text=self._active_source_text,
            )
            if not isinstance(result, dict):
                result = {"action": "escalate", "rationale": f"malformed review response: {result!r}"}
            revised_triple = None
            revised_payload = result.get("revised_triple")
            if result.get("action") == "revise" and isinstance(revised_payload, dict):
                revised_relation = self._canonicalize_schema_relation(
                    revised_payload.get("relation", triple.relation)
                )
                if revised_relation is None:
                    return GovernanceDecision(
                        triple=triple,
                        action="reject",
                        domain_id=assignment.primary_domain_id if assignment else None,
                        rationale=(
                            "Review proposed an out-of-schema revised relation "
                            f"'{revised_payload.get('relation')}' in fixed-schema mode; "
                            "rejecting to preserve the benchmark schema."
                        ),
                        assignment=assignment,
                    )
                revised_triple = Triple(
                    subject=revised_payload.get("subject", triple.subject),
                    relation=revised_relation,
                    object=revised_payload.get("object", triple.object),
                    confidence=triple.confidence,
                    source=triple.source,
                    metadata=triple.metadata,
                )
            return GovernanceDecision(
                triple=triple,
                action=result.get("action", "escalate"),
                domain_id=assignment.primary_domain_id if assignment else None,
                rationale=result.get("rationale", ""),
                revised_triple=revised_triple,
                assignment=assignment,
            )

        self.governed_kg.set_review_callback(review_callback)

    def _print_header(self) -> None:
        """Print orchestrator header."""
        print("\n" + "=" * 70)
        print("DELIBERATIVE MULTI-AGENT KNOWLEDGE GRAPH FRAMEWORK")
        print("=" * 70)
        print(f"Model Tiers:")
        for tier, model in self.model_tiers.items():
            print(f"  {tier.value}: {model}")
        print(f"\nFeatures:")
        print(f"  Self-Consistency: {'Enabled' if self.enable_self_consistency else 'Disabled'}")
        print(f"  Open-World Relations: {'Enabled' if self.enable_open_world else 'Disabled'}")
        print(f"  Cross-Document Resolution: {'Enabled' if self.enable_cross_document else 'Disabled'}")
        print(f"  Multi-Agent Deliberation: {'Enabled' if self.enable_deliberation else 'Disabled'}")
        print(f"  Governance Mode: {self.governance_mode}")
        print(f"  Skip Evidence Linking: {'Enabled' if self.skip_evidence_linking else 'Disabled'}")
        print(f"  Skip Verification: {'Enabled' if self.skip_verification else 'Disabled'}")
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
        self._active_source_text = context.text
        
        results = {}
        
        # ===== WORKER AGENTS =====
        
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
        if self.schema_override:
            # Use fixed schema instead of dynamic discovery
            domain_config = self.domain_classifier.build_fixed_schema(
                self.schema_override, context.document_id
            )
            domain_result_confidence = 0.95
            print(f"  Using fixed schema override ({len(domain_config.get('entity_types', []))} entity types, "
                  f"{len(domain_config.get('relation_types', []))} relation types)")
        elif self.reuse_corpus_schema and self._corpus_domain_config is not None:
            domain_config = self._corpus_domain_config
            domain_result_confidence = 0.95
            print(
                "  Reusing corpus schema "
                f"({len(domain_config.get('entity_types', []))} entity types, "
                f"{len(domain_config.get('relation_types', []))} relation types)"
            )
        else:
            domain_result = self.domain_classifier.run(context, segments=segments)
            domain_config = domain_result.items[0] if domain_result.items else {}
            domain_result_confidence = domain_result.confidence
            if self.reuse_corpus_schema and domain_config:
                self._corpus_domain_config = domain_config
        context.domain = domain_config.get("domain", "general")
        results["domain"] = context.domain
        print(f"  Domain: {context.domain} (confidence: {domain_result_confidence:.2f})")

        # Step 2b: Bootstrap preliminary domains from the discovered schema
        if self.debug_logger:
            self.debug_logger.log_stage_header("2b", "Governance Bootstrap")
        print("\n[2b/9] Governance Bootstrap")
        print("-" * 50)
        if not self.enable_governance or self.governed_kg is None:
            print("  Governance disabled - skipping")
        elif not self.governed_kg.org_chart.domains:
            preliminary_org = self._bootstrap_domains_from_schema(domain_config)
            self.governed_kg.set_org_chart(preliminary_org)
            print(f"  Bootstrapped {len(preliminary_org.domains)} preliminary domains")
        elif self.expand_org_chart_with_schema and domain_config:
            added, merged = self._expand_org_chart_from_schema(domain_config)
            print(
                "  Expanded governed org chart "
                f"(added {added}, merged {merged}, total {len(self.governed_kg.org_chart.domains)} domains)"
            )
        else:
            print(f"  Reusing existing governed org chart ({len(self.governed_kg.org_chart.domains)} domains)")
        
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

        print("\n[3b/9] Preliminary Domain Assignment")
        print("-" * 50)
        if not self.enable_governance or self.governed_kg is None:
            results["entities_domain_assigned"] = 0
            results["bootstrap_assignment_stats"] = {}
            print("  Governance disabled - skipping")
        else:
            entity_domain_assignments, bootstrap_stats = self._assign_entities_to_domains(entities)
            assigned_count = 0
            for entity in entities:
                entity_id = entity.get("id", entity.get("text", ""))
                candidate_domains = entity_domain_assignments.get(entity_id, [])
                if candidate_domains:
                    entity["candidate_domains"] = candidate_domains
                    assigned_count += 1
            results["entities_domain_assigned"] = assigned_count
            results["bootstrap_assignment_stats"] = bootstrap_stats
            self.governed_kg.set_bootstrap_assignment_stats(bootstrap_stats)
            print(f"  Assigned provisional domains to {assigned_count} entities")
        
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
        if relation_result.metadata.get("pairwise_pairs_considered"):
            print(
                "  Pairwise scoring:"
                f" pairs={relation_result.metadata['pairwise_pairs_considered']}"
                f", positives={relation_result.metadata.get('pairwise_positive_predictions', 0)}"
                f", added={relation_result.metadata.get('pairwise_triples_added', 0)}"
            )
        if relation_result.metadata.get("gleaned_triples_added"):
            print(f"  Gleaning pass added: {relation_result.metadata['gleaned_triples_added']} triples")
        if relation_result.metadata.get("funnel_diagnostics"):
            results["relation_funnel_diagnostics"] = relation_result.metadata["funnel_diagnostics"]

        # Step 4b: Connectivity Pass — find relations for disconnected entities
        connected_ids = set()
        for t in triples:
            connected_ids.add(t.get("subject_id") or t.get("subject", ""))
            connected_ids.add(t.get("object_id") or t.get("object", ""))
        disconnected_count = sum(
            1 for e in entities
            if (e.get("id", e.get("text", "")) not in connected_ids)
        )
        should_run_connectivity = (
            disconnected_count > 5
            and (
                self.enable_open_world
                or len(triples) <= 2
                or relation_result.confidence < 0.55
            )
        )
        if should_run_connectivity:
            if self.debug_logger:
                self.debug_logger.log_stage_header(4, "Connectivity Pass")
            print(f"\n[4b/9] Connectivity Pass ({disconnected_count} disconnected entities)")
            print("-" * 50)
            connectivity_relation_types = (
                relation_result.metadata.get("relation_types_found")
                or relation_result.metadata.get("relation_types_used")
                or relation_result.metadata.get("suggested_relation_types")
                or [
                    rt.get("type")
                    for rt in domain_config.get("relation_types", [])
                    if isinstance(rt, dict) and rt.get("type")
                ]
            )
            connectivity_triples = self.relation_extractor.extract_connectivity_relations(
                text=context.text,
                entities=entities,
                triples=triples,
                relation_types=connectivity_relation_types,
            )
            if connectivity_triples:
                triples.extend(connectivity_triples)
                context.relations = triples
                results["triples_extracted"] = len(triples)
                # Recount connected
                connected_after = set()
                for t in triples:
                    connected_after.add(t.get("subject_id") or t.get("subject", ""))
                    connected_after.add(t.get("object_id") or t.get("object", ""))
                disconnected_after = sum(
                    1 for e in entities
                    if (e.get("id", e.get("text", "")) not in connected_after)
                )
                print(f"  Total triples now: {len(triples)}")
                print(f"  Disconnected entities: {disconnected_count} → {disconnected_after}")
        else:
            print(f"\n[4b/9] Connectivity Pass — skipped ({disconnected_count} disconnected, threshold=5)")

        # Step 5: Evidence Linking
        if self.skip_evidence_linking:
            if self.debug_logger:
                self.debug_logger.log_stage_header(5, "Evidence Linking (Skipped)")
            print("\n[5/9] Evidence Linking — SKIPPED")
            print("-" * 50)
            linked_triples = triples
            results["triples_linked"] = len(linked_triples)
            print(f"  Passing through {len(linked_triples)} triples")
        else:
            if self.debug_logger:
                self.debug_logger.log_stage_header(5, "Evidence Linking")
            print("\n[5/9] Evidence Linking")
            print("-" * 50)
            evidence_result = self.evidence_linker.run(
                context,
                triples=triples,
                segments=segments,
                domain_config=domain_config,
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

        # Step 7: Extraction Validation (skip — consolidated into verification)
        if self.debug_logger:
            self.debug_logger.log_stage_header(7, "Extraction Validation (Skipped — consolidated)")
        print("\n[7/9] Extraction Validation (Skipped — consolidated into verification)")
        print("-" * 50)
        print("  Skipped: validation consolidated into verification step")
        results["refinement_iterations"] = 0

        # Step 8: Verification (single quality gate)
        if self.skip_verification:
            if self.debug_logger:
                self.debug_logger.log_stage_header(8, "Extraction Verification (Skipped)")
            print("\n[8/9] Extraction Verification — SKIPPED")
            print("-" * 50)
            verified = {"entities": entities, "approved_triples": linked_triples, "rejected_triples": []}
            results["approved_triples"] = len(linked_triples)
            results["rejected_triples"] = 0
            print(f"  Approved: {results['approved_triples']}")
            print(f"  Rejected: {results['rejected_triples']}")
        else:
            if self.debug_logger:
                self.debug_logger.log_stage_header(8, "Extraction Verification")
            print("\n[8/9] Extraction Verification")
            print("-" * 50)
            verification_result = self.verification_agent.run(
                context,
                entities=entities,
                triples=linked_triples,
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
        if self.governed_kg is not None and self.governed_kg.org_chart.domains:
            self.governed_kg.org_chart.refresh_memory_cards(self.governed_kg.kg)
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

    def _bootstrap_domains_from_schema(self, domain_config: Dict[str, Any]):
        """Create a preliminary org chart from domain-classifier output."""
        if self.governed_kg is None:
            return None
        if not domain_config:
            return self.governed_kg.org_chart
        return self.domain_builder.bootstrap_from_schema(domain_config)

    def _expand_org_chart_from_schema(self, domain_config: Dict[str, Any]) -> Tuple[int, int]:
        """Merge domains from a newly discovered schema into the current org chart."""
        if self.governed_kg is None or not domain_config:
            return 0, 0
        candidate_org = self.domain_builder.bootstrap_from_schema(domain_config)
        current = self.governed_kg.org_chart
        existing = {domain.domain_id: domain for domain in current.domains}
        added = 0
        merged = 0
        for candidate in candidate_org.domains:
            existing_domain = existing.get(candidate.domain_id)
            if existing_domain is None and current.domains:
                existing_domain = self._schema_merge_target(candidate, current.domains)
                if (
                    existing_domain is None
                    and self.target_num_domains
                    and len(current.domains) >= self.target_num_domains
                ):
                    existing_domain = self._best_domain_merge_target(candidate, current.domains)
            if existing_domain is None:
                current.domains.append(candidate)
                existing[candidate.domain_id] = candidate
                added += 1
                continue

            merged += 1
            self._merge_domain_schema(existing_domain, candidate)
        current._entity_domain_map_cache = None
        return added, merged

    @classmethod
    def _schema_merge_target(cls, candidate, domains):
        """Return an existing domain only when the new schema is genuinely related.

        Open-world schemas are discovered per document, so their domain IDs are
        unstable. Domain expansion must therefore be merge-first by semantic
        schema overlap, not append-unless-ID-matches.
        """
        best_domain = None
        best_score = 0.0
        for domain in domains:
            score = cls._domain_schema_similarity(candidate, domain)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain if best_score >= 0.28 else None

    @staticmethod
    def _merge_domain_schema(existing_domain, candidate) -> None:
        existing_domain.description = existing_domain.description or candidate.description
        existing_domain.relation_schema.update(candidate.relation_schema)

        existing_topics = {topic.topic_id for topic in existing_domain.topics}
        for topic in candidate.topics:
            if topic.topic_id not in existing_topics:
                existing_domain.topics.append(topic)
                existing_topics.add(topic.topic_id)

        existing_meta = existing_domain.metadata if isinstance(existing_domain.metadata, dict) else {}
        candidate_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        for key in ("seed_entity_types", "seed_relation_types"):
            values = list(existing_meta.get(key, []))
            for value in candidate_meta.get(key, []):
                if value not in values:
                    values.append(value)
            existing_meta[key] = values
        merged_ids = list(existing_meta.get("merged_domain_ids", []))
        if candidate.domain_id not in merged_ids and candidate.domain_id != existing_domain.domain_id:
            merged_ids.append(candidate.domain_id)
        existing_meta["merged_domain_ids"] = merged_ids
        existing_domain.metadata = existing_meta

    @classmethod
    def _domain_schema_similarity(cls, left, right) -> float:
        left_tokens = cls._domain_schema_tokens(left)
        right_tokens = cls._domain_schema_tokens(right)
        union = left_tokens | right_tokens
        token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0

        left_meta = left.metadata if isinstance(left.metadata, dict) else {}
        right_meta = right.metadata if isinstance(right.metadata, dict) else {}
        left_entities = cls._normalized_schema_set(left_meta.get("seed_entity_types", []))
        right_entities = cls._normalized_schema_set(right_meta.get("seed_entity_types", []))
        left_relations = cls._normalized_schema_set(
            list(left.relation_schema.keys()) + list(left_meta.get("seed_relation_types", []))
        )
        right_relations = cls._normalized_schema_set(
            list(right.relation_schema.keys()) + list(right_meta.get("seed_relation_types", []))
        )

        entity_score = cls._overlap_score(left_entities, right_entities)
        relation_score = cls._overlap_score(left_relations, right_relations)
        return max(token_score, entity_score * 0.75 + relation_score * 0.25, relation_score * 0.65 + token_score * 0.35)

    @staticmethod
    def _normalized_schema_set(values) -> set:
        import re

        normalized = set()
        for value in values or []:
            parts = re.findall(r"[a-z0-9]+", str(value).lower())
            if not parts:
                continue
            normalized.add("_".join(parts))
            for part in parts:
                if len(part) >= 3:
                    normalized.add(part.rstrip("s"))
        return normalized

    @staticmethod
    def _overlap_score(left: set, right: set) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / min(len(left), len(right))

    @classmethod
    def _domain_schema_tokens(cls, domain) -> set:
        metadata = domain.metadata if isinstance(domain.metadata, dict) else {}
        text = " ".join(
            [
                domain.domain_id,
                domain.label,
                domain.description,
                " ".join(metadata.get("seed_entity_types", [])),
                " ".join(metadata.get("seed_relation_types", [])),
                " ".join(domain.relation_schema.keys()),
                " ".join(topic.label for topic in getattr(domain, "topics", []) or []),
                " ".join(" ".join(topic.keywords) for topic in getattr(domain, "topics", []) or []),
            ]
        )
        import re

        tokens = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) >= 3:
                tokens.add(token.rstrip("s"))
        return tokens

    @staticmethod
    def _best_domain_merge_target(candidate, domains):
        """Choose the closest existing domain using only domain/schema text."""
        def tokens(domain) -> set:
            metadata = domain.metadata if isinstance(domain.metadata, dict) else {}
            text = " ".join(
                [
                    domain.domain_id,
                    domain.label,
                    domain.description,
                    " ".join(metadata.get("seed_entity_types", [])),
                    " ".join(metadata.get("seed_relation_types", [])),
                    " ".join(domain.relation_schema.keys()),
                ]
            )
            import re

            return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}

        candidate_tokens = tokens(candidate)
        best_domain = domains[0]
        best_score = -1.0
        for domain in domains:
            domain_tokens = tokens(domain)
            union = candidate_tokens | domain_tokens
            score = len(candidate_tokens & domain_tokens) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def _assign_entities_to_domains(
        self,
        entities: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Assign extracted entities to the preliminary domain structure."""
        if self.governed_kg is None or not self.governed_kg.org_chart.domains:
            return {}, {
                "num_entities": len(entities),
                "assigned_entities": 0,
                "unassigned_entities": len(entities),
                "multi_assigned_entities": 0,
                "assignment_coverage": 0.0,
            }
        return self.domain_builder.assign_entities_to_org_chart(
            entities,
            self.governed_kg.org_chart,
            return_diagnostics=True,
        )

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
        ENTITY_UNCERTAIN_LOW  = 0.15
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
        TRIPLE_UNCERTAIN_LOW  = 0.15
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
                subj_raw = triple.get("subject", "?")
                subj = subj_raw.get("name", "?") if isinstance(subj_raw, dict) else str(subj_raw)
                pred = triple.get("predicate", triple.get("relation", "?"))
                obj_raw = triple.get("object", "?")
                obj = obj_raw.get("name", "?") if isinstance(obj_raw, dict) else str(obj_raw)
                
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

    def _get_agent_by_name(self, name: str):
        """Get agent instance by name."""
        agent_map = {
            "EntityExtractor": self.entity_extractor,
            "RelationExtractor": self.relation_extractor,
            "EvidenceLinker": self.evidence_linker,
        }
        return agent_map.get(name)

    def _collect_votes_from_agents(
        self,
        hypothesis_id: str,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        author: str,
        context: Any,
    ) -> None:
        """Collect real votes from agents on a hypothesis.

        Calls each voting agent's evaluate_hypothesis_for_vote() method
        instead of fabricating votes with hardcoded if/else logic.
        """
        from multi_agent_kg.core.deliberation import VoteType

        voting_agents = ["EntityExtractor", "RelationExtractor", "EvidenceLinker"]
        for agent_name in voting_agents:
            if agent_name == author:
                continue  # don't self-vote
            agent = self._get_agent_by_name(agent_name)
            if agent is None:
                continue
            try:
                vote_type, confidence, rationale = agent.evaluate_hypothesis_for_vote(
                    hypothesis_content, hypothesis_type, context,
                )
                if vote_type != VoteType.ABSTAIN:
                    self.deliberation_coordinator.receive_vote(
                        hypothesis_id=hypothesis_id,
                        voter=agent_name,
                        vote_type=vote_type,
                        confidence=confidence,
                        rationale=rationale,
                    )
            except Exception as e:
                self.deliberation_coordinator.receive_vote(
                    hypothesis_id=hypothesis_id,
                    voter=agent_name,
                    vote_type=VoteType.ABSTAIN,
                    confidence=0.5,
                    rationale=f"Error during voting: {str(e)[:80]}",
                )

    def _collect_entity_votes(
        self,
        hypothesis_id: str,
        entity: Dict[str, Any],
        context: Any,
    ) -> None:
        """Collect real votes from agents on an entity hypothesis."""
        self._collect_votes_from_agents(
            hypothesis_id=hypothesis_id,
            hypothesis_content=entity,
            hypothesis_type="entity",
            author="EntityExtractor",
            context=context,
        )

    def _collect_triple_votes(
        self,
        hypothesis_id: str,
        triple: Dict[str, Any],
        context: Any,
    ) -> None:
        """Collect real votes from agents on a triple hypothesis."""
        self._collect_votes_from_agents(
            hypothesis_id=hypothesis_id,
            hypothesis_content=triple,
            hypothesis_type="triple",
            author="RelationExtractor",
            context=context,
        )

    def _run_hypothesis_debate(self, hypothesis_id: str) -> None:
        """Run a debate on a hypothesis using real agent evaluations.

        Each voting agent re-evaluates the hypothesis and provides a
        debate argument based on its own logic and memory.
        """
        from multi_agent_kg.core.deliberation import VoteType

        hypothesis = self.deliberation_coordinator.hypotheses.get(hypothesis_id)
        if not hypothesis:
            return

        voting_agents = ["EntityExtractor", "RelationExtractor", "EvidenceLinker"]
        for agent_name in voting_agents:
            agent = self._get_agent_by_name(agent_name)
            if agent is None:
                continue
            try:
                vote_type, confidence, rationale = agent.evaluate_hypothesis_for_vote(
                    hypothesis.content, hypothesis.hypothesis_type, None,
                )
                # Map vote to debate position
                if vote_type in (VoteType.STRONG_ACCEPT, VoteType.ACCEPT, VoteType.WEAK_ACCEPT):
                    position = "support"
                elif vote_type in (VoteType.REJECT, VoteType.STRONG_REJECT, VoteType.WEAK_REJECT):
                    position = "oppose"
                else:
                    continue  # skip abstentions in debate

                self.deliberation_coordinator.receive_debate_argument(
                    hypothesis_id=hypothesis_id,
                    agent=agent_name,
                    position=position,
                    argument=f"[{vote_type.value}, conf={confidence:.2f}] {rationale}",
                )
            except Exception:
                pass

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
        failed_documents = []

        for i, doc in enumerate(documents):
            print(f"\n[Document {i+1}/{len(documents)}]")
            try:
                result = self.process_document(
                    text=doc.get("text"),
                    source_path=doc.get("source"),
                    document_id=doc.get("id"),
                    metadata=doc.get("metadata"),
                )
                all_results.append(result)
            except Exception as exc:
                failed = {
                    "document_id": doc.get("id"),
                    "error": str(exc),
                }
                failed_documents.append(failed)
                print(f"  ERROR: {failed['document_id']} failed: {failed['error']}")
                if not self.continue_on_document_error:
                    raise

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
            "failed_documents": failed_documents,
            "relation_funnel_summary": self.get_relation_funnel_summary(),
        }
        
        print("\n" + "=" * 70)
        print("CORPUS PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Documents: {aggregate['documents_processed']}")
        print(f"Total Entities: {aggregate['total_entities']}")
        print(f"Total Triples: {aggregate['total_triples']}")
        print(f"Total Time: {aggregate['total_time']:.2f}s")
        print(f"Failed Documents: {len(failed_documents)}")
        print("=" * 70 + "\n")
        
        return aggregate

    def get_relation_funnel_summary(self) -> Dict[str, Any]:
        """Aggregate relation-extraction funnel diagnostics across processed docs."""
        numeric_fields = [
            "segments_processed",
            "entities_seen",
            "relations_found",
            "head_bindings",
            "tail_triples",
            "pairwise_pairs_considered",
            "pairwise_positive_predictions",
            "pairwise_triples_added",
            "gleaned_triples_added",
            "invalid_self_refs_filtered",
            "post_alignment_triples",
            "post_dedupe_triples",
            "final_triples",
        ]
        totals: Dict[str, Any] = {field: 0 for field in numeric_fields}
        docs_with_diagnostics = 0
        docs_with_zero_final_triples: List[str] = []
        docs_with_stage1_relations_but_no_final_triples: List[str] = []
        per_doc: List[Dict[str, Any]] = []

        for entry in self.processing_history:
            results = entry.get("results", {}) if isinstance(entry, dict) else {}
            diagnostics = results.get("relation_funnel_diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            docs_with_diagnostics += 1
            doc_id = str(entry.get("document_id") or diagnostics.get("document_id") or "")
            row = {"document_id": doc_id}
            for field in numeric_fields:
                value = diagnostics.get(field, 0)
                value = value if isinstance(value, (int, float)) else 0
                totals[field] += value
                row[field] = value
            if row["final_triples"] == 0:
                docs_with_zero_final_triples.append(doc_id)
            if row["relations_found"] > 0 and row["final_triples"] == 0:
                docs_with_stage1_relations_but_no_final_triples.append(doc_id)
            per_doc.append(row)

        summary = {
            **totals,
            "documents_with_diagnostics": docs_with_diagnostics,
            "docs_with_zero_final_triples": docs_with_zero_final_triples,
            "docs_with_stage1_relations_but_no_final_triples": docs_with_stage1_relations_but_no_final_triples,
            "per_doc": per_doc,
        }
        if totals["relations_found"]:
            summary["tail_triples_per_stage1_relation"] = round(
                totals["tail_triples"] / totals["relations_found"], 4
            )
        else:
            summary["tail_triples_per_stage1_relation"] = 0.0
        if totals["post_alignment_triples"]:
            summary["dedupe_retention"] = round(
                totals["post_dedupe_triples"] / totals["post_alignment_triples"], 4
            )
        else:
            summary["dedupe_retention"] = 0.0
        return summary

    def _resolve_cross_document_entities(self) -> None:
        """Resolve entities across documents using fuzzy matching.

        Uses find_entity_matches() from kg_operations to detect duplicate
        entities, registers aliases in SharedMemory, and remaps triples
        to canonical entity IDs.
        """
        from multi_agent_kg.core.kg_operations import find_entity_matches

        kg = self.knowledge_graph
        entities = kg.entities

        if len(entities) < 2:
            print("  Not enough entities for cross-document resolution")
            return

        # Find entity matches (self-match to detect duplicates within the KG)
        # Split entities into groups by document to compare across docs
        matches = find_entity_matches(entities, entities, threshold=0.80)

        aliases_registered = 0
        remapped_triples = 0

        for source_id, target_id in matches.items():
            if source_id == target_id:
                continue  # skip self-matches

            # Register alias in SharedMemory
            self.shared_memory.register_entity_alias(source_id, target_id)
            aliases_registered += 1

        # Remap triples to use canonical entity IDs
        alias_map = self.shared_memory.entity_aliases
        if alias_map:
            for triple in kg.triples:
                new_subj = alias_map.get(triple.subject)
                new_obj = alias_map.get(triple.object)
                if new_subj and new_subj != triple.subject:
                    triple.subject = new_subj
                    remapped_triples += 1
                if new_obj and new_obj != triple.object:
                    triple.object = new_obj
                    remapped_triples += 1

        stats = self.shared_memory.get_stats()
        print(f"  Entity aliases registered: {stats.get('entity_aliases', 0)}")
        print(f"  New aliases from resolution: {aliases_registered}")
        print(f"  Triples remapped: {remapped_triples}")
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
