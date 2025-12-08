"""
Advanced Orchestrator with Multi-Document Context and Agent Collaboration.

Features:
- Maintains context across multiple documents
- Enables agent-to-agent communication
- Supports iterative refinement loops
- Implements voting and consensus mechanisms
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple, Conflict
from multi_agent_kg.core.memory import SharedMemory, MemoryType
from multi_agent_kg.core.communication import MessageBus, CollaborationProtocol, CommunicationType
from multi_agent_kg.core.config import LLMConfig

from multi_agent_kg.agents.ingestion_agent import IngestionAgent
from multi_agent_kg.agents.segmenter_agent import SegmenterAgent
from multi_agent_kg.agents.summarizer_agent import SummarizerAgent
from multi_agent_kg.agents.entity_agent import EntityAgent
from multi_agent_kg.agents.open_world_relation_agent import OpenWorldRelationAgent
from multi_agent_kg.agents.conflict_agent import ConflictAgent
from multi_agent_kg.agents.verifier_agent import VerifierAgent


class AdvancedOrchestrator:
    """
    Advanced orchestrator for multi-agent knowledge graph construction.
    
    Key Features:
    1. Multi-document context maintenance
    2. Open-world relation discovery
    3. Agent communication and collaboration
    4. Iterative refinement loops
    5. Cross-document entity resolution
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        enable_summarization: bool = False,
        enable_verification: bool = True,
        enable_collaboration: bool = True,
        confidence_threshold: float = 0.5,
        refinement_rounds: int = 1,
    ):
        """
        Initialize the advanced orchestrator.
        
        Args:
            llm_config: LLM configuration
            knowledge_graph: Existing KG or create new
            enable_summarization: Use summarization agent
            enable_verification: Use verification agent
            enable_collaboration: Enable agent communication
            confidence_threshold: Minimum confidence for triples
            refinement_rounds: Number of refinement iterations
        """
        self.llm_config = llm_config or LLMConfig(model="gpt-3.5-turbo")
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.enable_summarization = enable_summarization
        self.enable_verification = enable_verification
        self.enable_collaboration = enable_collaboration
        self.confidence_threshold = confidence_threshold
        self.refinement_rounds = refinement_rounds
        
        # Shared infrastructure
        self.shared_memory = SharedMemory()
        self.message_bus = MessageBus() if enable_collaboration else None
        self.collab = CollaborationProtocol(self.message_bus) if self.message_bus else None
        
        # Initialize agents with shared infrastructure
        self._init_agents()
        
        # Document tracking
        self.document_count = 0
        self.session_start = datetime.now()
        
        self._print_header()

    def _init_agents(self) -> None:
        """Initialize all agents with shared memory and message bus."""
        
        self.ingestion_agent = IngestionAgent(
            knowledge_graph=self.knowledge_graph,
        )
        
        self.segmenter_agent = SegmenterAgent(
            knowledge_graph=self.knowledge_graph,
            min_segment_length=50,
        )
        
        self.summarizer_agent = SummarizerAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
        )
        
        self.entity_agent = EntityAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
        )
        
        # Use open-world relation agent
        self.relation_agent = OpenWorldRelationAgent(
            knowledge_graph=self.knowledge_graph,
            shared_memory=self.shared_memory,
            message_bus=self.message_bus,
            llm_config=self.llm_config,
        )
        
        self.conflict_agent = ConflictAgent(
            knowledge_graph=self.knowledge_graph,
        )
        
        self.verifier_agent = VerifierAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
            confidence_threshold=self.confidence_threshold,
        )

    def _print_header(self) -> None:
        """Print orchestrator header."""
        print("ADVANCED MULTI-AGENT KNOWLEDGE GRAPH FRAMEWORK")
        print(f"Mode: OPEN-WORLD (No Predefined Schema)")
        print(f"LLM Model: {self.llm_config.model}")
        print(f"Multi-Document Context: Enabled")
        print(f"Agent Collaboration: {'Enabled' if self.enable_collaboration else 'Disabled'}")
        print(f"Refinement Rounds: {self.refinement_rounds}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print()

    def process_document(
        self,
        document_source: Optional[str] = None,
        text: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        """
        Process a single document through the pipeline.
        
        Args:
            document_source: Path to document file
            text: Raw text content
            document_id: Optional ID for tracking
            metadata: Optional document metadata
            
        Returns:
            Updated knowledge graph
        """
        self.document_count += 1
        
        # Generate document ID if not provided
        if not document_id:
            content = text or document_source or ""
            doc_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            document_id = f"doc_{self.document_count}_{doc_hash}"
        
        print(f"\nProcessing Document: {document_id}")
        
        # Step 1: Ingestion
        print("\nStep 1: Document Ingestion")
        raw_text = self.ingestion_agent.run(source=document_source, text=text)
        print(f"   Loaded {len(raw_text)} characters")
        
        # Register document
        self.shared_memory.register_document(document_id, raw_text, metadata)
        
        # Step 2: Segmentation
        print("\nStep 2: Text Segmentation")
        segments = self.segmenter_agent.run(text=raw_text)
        print(f"   Created {len(segments)} segments")
        
        # Step 3: Summarization (optional)
        text_for_extraction = segments
        if self.enable_summarization:
            print("\nStep 3: Summarization")
            summaries = self.summarizer_agent.run(segments=segments)
            text_for_extraction = summaries
            print(f"   Generated {len(summaries)} summaries")
        
        # Step 4: Entity Extraction
        print("\nStep 4: Entity Extraction")
        entities = self.entity_agent.run(text_chunks=text_for_extraction)
        print(f"   Extracted {len(entities)} entities")
        self._print_entities(entities)
        
        # Store entities in memory
        self._store_entities_in_memory(entities, document_id)
        
        # Step 5: Open-World Relation Extraction
        print("\nStep 5: Open-World Relation Extraction")
        triples = self.relation_agent.run(
            text_chunks=segments,
            entities=entities,
            document_id=document_id,
            use_context=True,
            propose_to_agents=self.enable_collaboration,
        )
        print(f"   Extracted {len(triples)} triples")
        self._print_triples(triples)
        
        # Show discovered relations
        self._print_discovered_relations()
        
        # Step 6: Refinement Loop (if enabled)
        if self.refinement_rounds > 1:
            print(f"\nStep 6: Iterative Refinement ({self.refinement_rounds} rounds)")
            triples = self._refinement_loop(triples, segments, entities, raw_text)
        
        # Step 7: Conflict Detection
        print("\nStep 7: Conflict Detection")
        safe_triples, conflicts = self.conflict_agent.run(candidate_triples=triples)
        if conflicts:
            print(f"   Found {len(conflicts)} conflicts")
            for c in conflicts[:3]:
                print(f"      - {c}")
        else:
            print("   No conflicts detected")
        
        # Step 8: Verification (optional)
        approved_triples = safe_triples
        if self.enable_verification:
            print("\nStep 8: Triple Verification")
            approved_triples = self.verifier_agent.run(
                triples=safe_triples,
                source_text=raw_text,
            )
            print(f"   {len(approved_triples)} triples verified")
        
        # Step 9: Knowledge Graph Integration
        print("\nStep 9: Knowledge Graph Integration")
        self._integrate_triples(approved_triples, entities, document_id)
        print(f"   Integrated {len(approved_triples)} triples")
        
        # Update entity contexts
        self._update_entity_contexts(entities, approved_triples, document_id)
        
        self._print_summary()
        
        return self.knowledge_graph

    def process_corpus(
        self,
        documents: List[Dict[str, Any]],
        resolve_cross_document_entities: bool = True,
    ) -> KnowledgeGraph:
        """
        Process multiple documents as a corpus.
        
        Args:
            documents: List of dicts with 'text' and optional 'id', 'metadata'
            resolve_cross_document_entities: Enable cross-document entity resolution
            
        Returns:
            Knowledge graph with all extracted knowledge
        """
        print("\n")
        print(f"PROCESSING CORPUS: {len(documents)} documents")
        print()
        
        for i, doc in enumerate(documents):
            print()
            print(f"Document {i+1}/{len(documents)}")
            
            self.process_document(
                text=doc.get("text"),
                document_source=doc.get("source"),
                document_id=doc.get("id"),
                metadata=doc.get("metadata"),
            )
        
        # Cross-document entity resolution
        if resolve_cross_document_entities:
            print("\nCross-Document Entity Resolution")
            self._resolve_entities_across_documents()
        
        self._print_final_stats()
        
        return self.knowledge_graph

    def _refinement_loop(
        self,
        triples: List[Triple],
        segments: List[str],
        entities: List[Entity],
        source_text: str,
    ) -> List[Triple]:
        """
        Run iterative refinement on extracted triples.
        """
        current_triples = triples
        
        for round_num in range(1, self.refinement_rounds):
            print(f"   Round {round_num+1}: Refining {len(current_triples)} triples...")
            
            # Get feedback from other agents via message bus
            if self.message_bus:
                # Post current triples for review
                for triple in current_triples[:10]:  # Limit for efficiency
                    self.collab.propose_hypothesis(
                        proposer="Orchestrator",
                        hypothesis={
                            "subject": triple.subject,
                            "relation": triple.relation,
                            "object": triple.object,
                        },
                        confidence=triple.confidence or 0.7,
                    )
            
            # For now, just validate existing triples
            # In a full implementation, agents would provide feedback
            
        return current_triples

    def _integrate_triples(
        self,
        triples: List[Triple],
        entities: List[Entity],
        document_id: str,
    ) -> None:
        """Integrate approved triples into the knowledge graph."""
        
        # First ensure all entities exist
        for entity in entities:
            self.knowledge_graph.add_entity(
                entity_id=entity.id,
                labels=entity.labels,
                entity_type=entity.type,
                metadata={
                    "source_document": document_id,
                    **entity.metadata,
                },
            )
        
        # Add triples
        for triple in triples:
            self.knowledge_graph.add_triple(
                subject=triple.subject,
                relation=triple.relation,
                obj=triple.object,
                confidence=triple.confidence,
                source=f"{triple.source}:{document_id}",
                metadata=triple.metadata,
            )

    def _store_entities_in_memory(
        self,
        entities: List[Entity],
        document_id: str,
    ) -> None:
        """Store extracted entities in shared memory."""
        self.shared_memory.store(
            memory_type=MemoryType.EPISODIC,
            content={
                "entities": [{"id": e.id, "type": e.type, "labels": e.labels} for e in entities],
                "document_id": document_id,
            },
            source="EntityAgent",
        )

    def _update_entity_contexts(
        self,
        entities: List[Entity],
        triples: List[Triple],
        document_id: str,
    ) -> None:
        """Update entity contexts in shared memory."""
        entity_triples: Dict[str, List[Dict]] = {}
        
        for triple in triples:
            for entity_id in [triple.subject, triple.object]:
                if entity_id not in entity_triples:
                    entity_triples[entity_id] = []
                entity_triples[entity_id].append({
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                    "role": "subject" if entity_id == triple.subject else "object",
                })
        
        for entity_id, relations in entity_triples.items():
            self.shared_memory.add_entity_context(
                entity_id,
                {
                    "document_id": document_id,
                    "relations": relations,
                },
            )

    def _resolve_entities_across_documents(self) -> None:
        """Resolve entities across multiple documents."""
        # Simple approach: find similar entity names
        entity_ids = list(self.knowledge_graph.entities.keys())
        
        resolved_count = 0
        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i+1:]:
                # Check for potential matches
                if self._entities_match(id1, id2):
                    # Register as alias
                    canonical = id1 if len(id1) >= len(id2) else id2
                    alias = id2 if canonical == id1 else id1
                    self.shared_memory.register_entity_alias(alias, canonical)
                    resolved_count += 1
        
        print(f"   Resolved {resolved_count} entity aliases")

    def _entities_match(self, id1: str, id2: str) -> bool:
        """Check if two entity IDs might refer to the same entity."""
        # Simple heuristics
        id1_lower = id1.lower().strip()
        id2_lower = id2.lower().strip()
        
        # Exact match
        if id1_lower == id2_lower:
            return True
        
        # One contains the other
        if id1_lower in id2_lower or id2_lower in id1_lower:
            return True
        
        # Check labels
        e1 = self.knowledge_graph.entities.get(id1)
        e2 = self.knowledge_graph.entities.get(id2)
        
        if e1 and e2:
            labels1 = {l.lower() for l in e1.labels}
            labels2 = {l.lower() for l in e2.labels}
            
            if labels1 & labels2:
                return True
            
            if id1_lower in labels2 or id2_lower in labels1:
                return True
        
        return False

    def _print_entities(self, entities: List[Entity], max_show: int = 8) -> None:
        """Print extracted entities."""
        for e in entities[:max_show]:
            type_str = f" ({e.type})" if e.type else ""
            print(f"      - {e.id}{type_str}")
        if len(entities) > max_show:
            print(f"      ... and {len(entities) - max_show} more")

    def _print_triples(self, triples: List[Triple], max_show: int = 8) -> None:
        """Print extracted triples."""
        for t in triples[:max_show]:
            conf = f" [{t.confidence:.2f}]" if t.confidence else ""
            print(f"      - ({t.subject}) -[{t.relation}]-> ({t.object}){conf}")
        if len(triples) > max_show:
            print(f"      ... and {len(triples) - max_show} more")

    def _print_discovered_relations(self) -> None:
        """Print discovered relation types."""
        relations = self.relation_agent.get_relation_ontology()
        if relations:
            print(f"\n   Discovered Relation Types ({len(relations)}):")
            for name, info in sorted(relations.items(), key=lambda x: x[1]["frequency"], reverse=True)[:10]:
                print(f"      - {name}: {info['definition'][:50]}... (freq: {info['frequency']})")

    def _print_summary(self) -> None:
        """Print document processing summary."""
        stats = self.knowledge_graph.get_stats()
        print()
        print(f"Document Summary:")
        print(f"   Entities: {stats['num_entities']}")
        print(f"   Triples: {stats['num_triples']}")
        print()

    def _print_final_stats(self) -> None:
        """Print final corpus statistics."""
        kg_stats = self.knowledge_graph.get_stats()
        mem_stats = self.shared_memory.get_stats()
        
        print()
        print("FINAL CORPUS STATISTICS")
        
        print(f"\nKnowledge Graph:")
        print(f"   Total Entities: {kg_stats['num_entities']}")
        print(f"   Total Triples: {kg_stats['num_triples']}")
        
        if kg_stats['entity_type_counts']:
            print(f"\n   Entity Types:")
            for etype, count in sorted(kg_stats['entity_type_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      - {etype}: {count}")
        
        if kg_stats['relation_counts']:
            print(f"\n   Relation Types:")
            for rel, count in sorted(kg_stats['relation_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      - {rel}: {count}")
        
        print(f"\nShared Memory:")
        print(f"   Total Memories: {mem_stats['total_memories']}")
        print(f"   Documents Processed: {mem_stats['documents_processed']}")
        print(f"   Unique Entities Tracked: {mem_stats['unique_entities']}")
        print(f"   Entity Aliases: {mem_stats['entity_aliases']}")
        print()

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Get the current knowledge graph."""
        return self.knowledge_graph

    def get_memory(self) -> SharedMemory:
        """Get the shared memory system."""
        return self.shared_memory

    def get_discovered_ontology(self) -> Dict[str, Any]:
        """Get the discovered relation ontology."""
        return self.relation_agent.get_relation_ontology()

    def export_all(self, base_path: str) -> None:
        """Export all data to files."""
        import json
        
        # Export knowledge graph
        kg_path = f"{base_path}_knowledge_graph.json"
        with open(kg_path, "w") as f:
            f.write(self.knowledge_graph.to_json())
        print(f"Knowledge graph exported to {kg_path}")
        
        # Export memory
        mem_path = f"{base_path}_memory.json"
        with open(mem_path, "w") as f:
            json.dump(self.shared_memory.export(), f, indent=2, default=str)
        print(f"Memory exported to {mem_path}")
        
        # Export ontology
        ont_path = f"{base_path}_ontology.json"
        with open(ont_path, "w") as f:
            json.dump(self.get_discovered_ontology(), f, indent=2)
        print(f"Ontology exported to {ont_path}")
