"""
Deliberative Multi-Agent Orchestrator.

This orchestrator implements the full integrated pipeline with:
- SharedMemory for cross-document context and blackboard voting
- MessageBus for inter-agent communication
- Tiered model selection
- Iterative refinement with quality thresholds
- Escalation and deliberation mechanisms

Architecture:
  Workers: DocumentProcessor -> DomainClassifier -> EntityExtractor -> 
           RelationExtractor -> EvidenceLinker
  Coordinators: ExtractionValidator -> ExtractionVerificationAgent -> 
                KnowledgeOrganizer

Novel Features:
- Blackboard pattern for hypothesis voting
- Self-consistency for confidence estimation
- Cross-document entity resolution
- Open-world relation discovery
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CollaborationProtocol
from multi_agent_kg.core.config import LLMConfig

from multi_agent_kg.agents.base import AgentContext, ModelTier
from multi_agent_kg.agents.document_processor import DocumentProcessor
from multi_agent_kg.agents.domain_classifier import DomainClassifier
from multi_agent_kg.agents.entity_extractor import EntityExtractor
from multi_agent_kg.agents.relation_extractor import RelationExtractor
from multi_agent_kg.agents.evidence_linker import EvidenceLinker
from multi_agent_kg.agents.extraction_validator import ExtractionValidator
from multi_agent_kg.agents.extraction_verification_agent import ExtractionVerificationAgent
from multi_agent_kg.agents.knowledge_organizer import KnowledgeOrganizer


class DeliberativeOrchestrator:
    """
    Deliberative Multi-Agent Orchestrator for Knowledge Graph Construction.
    
    This orchestrator coordinates 8 agents in a tiered pipeline:
    
    Worker Agents (extraction):
    1. DocumentProcessor: Ingests and segments documents
    2. DomainClassifier: Classifies domain for tailored extraction
    3. EntityExtractor: Multi-stage entity extraction with self-consistency
    4. RelationExtractor: RHF-style relation extraction with open-world support
    5. EvidenceLinker: Links triples to source evidence
    
    Coordinator Agents (validation):
    6. ExtractionValidator: Validates and refines extractions (max 4 iterations)
    7. ExtractionVerificationAgent: Final verification against source
    8. KnowledgeOrganizer: Integrates into knowledge graph
    
    Novel Features:
    - SharedMemory: Episodic, semantic, working memory + blackboard pattern
    - MessageBus: Inter-agent communication for escalation and feedback
    - Self-Consistency: Multiple LLM samples for confidence estimation
    - Cross-Document: Entity resolution across multiple documents
    - Open-World: Discovery of new relation types
    
    Quality Assurance:
    - Iterative refinement up to 4 iterations
    - Quality threshold of 0.85 for acceptance
    - Blackboard voting for ambiguous cases
    - Escalation to coordinators for low-confidence items
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        quality_threshold: float = 0.85,
        max_refinement_iterations: int = 4,
        enable_self_consistency: bool = True,
        enable_open_world: bool = True,
        enable_cross_document: bool = True,
        model_tiers: Optional[Dict[ModelTier, str]] = None,
    ):
        """
        Initialize the deliberative orchestrator.
        
        Args:
            llm_config: Base LLM configuration
            knowledge_graph: Existing KG or creates new
            quality_threshold: Minimum confidence for acceptance (default 0.85)
            max_refinement_iterations: Max refinement loops (default 4)
            enable_self_consistency: Use self-consistency for confidence
            enable_open_world: Allow discovery of new relation types
            enable_cross_document: Enable cross-document entity resolution
            model_tiers: Custom model tier mapping
        """
        self.llm_config = llm_config or LLMConfig()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.quality_threshold = quality_threshold
        self.max_refinement_iterations = max_refinement_iterations
        self.enable_self_consistency = enable_self_consistency
        self.enable_open_world = enable_open_world
        self.enable_cross_document = enable_cross_document
        
        # Model tier configuration
        self.model_tiers = model_tiers or {
            ModelTier.SMALL: "gpt-3.5-turbo",
            ModelTier.MEDIUM: "gpt-4o-mini",
            ModelTier.LARGE: "gpt-4o",
        }
        
        # Shared infrastructure
        self.shared_memory = SharedMemory()
        self.message_bus = MessageBus()
        self.collab = CollaborationProtocol(self.message_bus)
        
        # Initialize agents
        self._init_agents()
        
        # Tracking
        self.document_count = 0
        self.session_start = datetime.now()
        self.processing_history = []
        
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
        print(f"\nQuality Settings:")
        print(f"  Threshold: {self.quality_threshold}")
        print(f"  Max Refinement Iterations: {self.max_refinement_iterations}")
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
        
        # ===== WORKER AGENTS =====
        
        # Step 1: Document Processing
        print("\n[1/8] Document Processing")
        print("-" * 50)
        doc_result = self.document_processor.run(context, source_path=source_path)
        segments = doc_result.items
        results["segments"] = len(segments)
        print(f"  Segments: {len(segments)}")
        
        # Step 2: Domain Classification
        print("\n[2/8] Domain Classification")
        print("-" * 50)
        domain_result = self.domain_classifier.run(context, segments=segments)
        domain_config = domain_result.items[0] if domain_result.items else {}
        context.domain = domain_config.get("domain", "general")
        results["domain"] = context.domain
        print(f"  Domain: {context.domain} (confidence: {domain_result.confidence:.2f})")
        
        # Step 3: Entity Extraction
        print("\n[3/8] Entity Extraction (Multi-Stage)")
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
        print("\n[4/8] Relation Extraction (RHF Pipeline)")
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
        
        # Step 5: Evidence Linking
        print("\n[5/8] Evidence Linking")
        print("-" * 50)
        evidence_result = self.evidence_linker.run(
            context,
            triples=triples,
            segments=segments,
        )
        linked_triples = evidence_result.items
        results["triples_linked"] = len(linked_triples)
        print(f"  Linked: {len(linked_triples)} (confidence: {evidence_result.confidence:.2f})")
        
        # ===== COORDINATOR AGENTS =====
        
        # Step 6: Extraction Validation
        print("\n[6/8] Extraction Validation (Iterative Refinement)")
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
        
        # Step 7: Verification
        print("\n[7/8] Extraction Verification")
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
        
        # Step 8: Knowledge Organization
        print("\n[8/8] Knowledge Graph Integration")
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

    def export(self) -> Dict[str, Any]:
        """Export complete system state."""
        return {
            "knowledge_graph": self.knowledge_organizer.export_knowledge_graph(),
            "memory": self.shared_memory.export(),
            "stats": self.get_stats(),
            "processing_history": self.processing_history,
        }
