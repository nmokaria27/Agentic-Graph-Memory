"""
Orchestrator for coordinating the multi-agent knowledge graph enrichment pipeline.
"""

from typing import Optional, Dict, Any
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.config import RelationSchema, LLMConfig
from multi_agent_kg.agents.ingestion_agent import IngestionAgent
from multi_agent_kg.agents.segmenter_agent import SegmenterAgent
from multi_agent_kg.agents.summarizer_agent import SummarizerAgent
from multi_agent_kg.agents.entity_agent import EntityAgent
from multi_agent_kg.agents.relation_agent import RelationAgent
from multi_agent_kg.agents.schema_agent import SchemaAgent
from multi_agent_kg.agents.conflict_agent import ConflictAgent
from multi_agent_kg.agents.verifier_agent import VerifierAgent


class Orchestrator:
    """
    Orchestrates the multi-agent workflow for knowledge graph construction.

    Pipeline:
    1. Ingestion: Load document
    2. Segmentation: Split into chunks
    3. Summarization: Summarize chunks (optional)
    4. Entity Extraction: Extract entities
    5. Relation Extraction: Extract relations
    6. Schema Alignment: Validate against schema
    7. Conflict Detection: Find conflicts
    8. Verification: Verify triples
    9. Integration: Add to knowledge graph
    """

    def __init__(
        self,
        relation_schema: Optional[RelationSchema] = None,
        llm_config: Optional[LLMConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        enable_summarization: bool = True,
        enable_verification: bool = True,
        min_segment_length: int = 50,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the orchestrator.

        Args:
            relation_schema: Schema defining allowed relation types
            llm_config: Configuration for LLM calls
            knowledge_graph: Existing knowledge graph (or create new one)
            enable_summarization: Whether to use summarization agent
            enable_verification: Whether to use verification agent
            min_segment_length: Minimum length for text segments
            confidence_threshold: Minimum confidence for triple approval
        """
        self.relation_schema = relation_schema or RelationSchema()
        self.llm_config = llm_config or LLMConfig()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.enable_summarization = enable_summarization
        self.enable_verification = enable_verification

        # Initialize agents
        self.ingestion_agent = IngestionAgent(
            knowledge_graph=self.knowledge_graph,
        )

        self.segmenter_agent = SegmenterAgent(
            knowledge_graph=self.knowledge_graph,
            min_segment_length=min_segment_length,
        )

        self.summarizer_agent = SummarizerAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
        )

        self.entity_agent = EntityAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
        )

        self.relation_agent = RelationAgent(
            knowledge_graph=self.knowledge_graph,
            relation_schema=self.relation_schema,
            llm_config=self.llm_config,
        )

        self.schema_agent = SchemaAgent(
            knowledge_graph=self.knowledge_graph,
            relation_schema=self.relation_schema,
        )

        self.conflict_agent = ConflictAgent(
            knowledge_graph=self.knowledge_graph,
        )

        self.verifier_agent = VerifierAgent(
            knowledge_graph=self.knowledge_graph,
            llm_config=self.llm_config,
            confidence_threshold=confidence_threshold,
        )

        self._print_header()

    def _print_header(self) -> None:
        """Print orchestrator header."""
        print("\n" + "=" * 70)
        print("MULTI-AGENT KNOWLEDGE GRAPH ENRICHMENT FRAMEWORK")
        print("=" * 70)
        print(f"Relation Schema: {len(self.relation_schema.types)} types")
        print(f"LLM Model: {self.llm_config.model}")
        print(f"Summarization: {'Enabled' if self.enable_summarization else 'Disabled'}")
        print(f"Verification: {'Enabled' if self.enable_verification else 'Disabled'}")
        print("=" * 70 + "\n")

    def process_document(
        self,
        document_source: Optional[str] = None,
        text: Optional[str] = None,
        open_world: bool = False,
    ) -> KnowledgeGraph:
        """
        Process a document through the complete multi-agent pipeline.

        Args:
            document_source: Path to document file
            text: Raw text content (if no file path)
            open_world: Allow discovery of new relation types

        Returns:
            Updated knowledge graph

        Raises:
            ValueError: If neither document_source nor text is provided
        """
        print("\n🚀 Starting knowledge graph enrichment pipeline...\n")

        # Step 1: Ingestion
        print("📄 Step 1: Ingestion")
        print("-" * 70)
        raw_text = self.ingestion_agent.run(source=document_source, text=text)
        print(f"✓ Loaded {len(raw_text)} characters\n")

        # Step 2: Segmentation
        print("✂️  Step 2: Segmentation")
        print("-" * 70)
        segments = self.segmenter_agent.run(text=raw_text)
        print(f"✓ Created {len(segments)} segments\n")

        # Step 3: Summarization (optional)
        text_for_extraction = segments
        if self.enable_summarization:
            print("📝 Step 3: Summarization")
            print("-" * 70)
            summaries = self.summarizer_agent.run(segments=segments)
            text_for_extraction = summaries
            print(f"✓ Generated {len(summaries)} summaries\n")
        else:
            print("⏭️  Step 3: Summarization (skipped)\n")

        # Step 4: Entity Extraction
        print("🏷️  Step 4: Entity Extraction")
        print("-" * 70)
        entities = self.entity_agent.run(text_chunks=text_for_extraction)
        print(f"✓ Extracted {len(entities)} unique entities")
        self._print_entities_preview(entities)
        print()

        # Step 5: Relation Extraction
        print("🔗 Step 5: Relation Extraction")
        print("-" * 70)
        candidate_triples = self.relation_agent.run(
            text_chunks=segments,  # Use original segments for better context
            entities=entities,
            open_world=open_world,
        )
        print(f"✓ Extracted {len(candidate_triples)} candidate triples")
        self._print_triples_preview(candidate_triples)
        print()

        # Step 6: Schema Alignment
        print("🔍 Step 6: Schema Alignment")
        print("-" * 70)
        aligned_triples = self.schema_agent.run(
            entities=entities,
            triples=candidate_triples,
        )
        print(f"✓ {len(aligned_triples)} triples aligned with schema\n")

        # Step 7: Conflict Detection
        print("⚠️  Step 7: Conflict Detection")
        print("-" * 70)
        non_conflicting_triples, conflicts = self.conflict_agent.run(
            candidate_triples=aligned_triples,
        )
        if conflicts:
            print(f"⚠️  Found {len(conflicts)} conflicts")
            for conflict in conflicts[:3]:  # Show first 3
                print(f"  - {conflict}")
            if len(conflicts) > 3:
                print(f"  ... and {len(conflicts) - 3} more")
        else:
            print("✓ No conflicts detected")
        print(f"✓ {len(non_conflicting_triples)} non-conflicting triples\n")

        # Step 8: Verification
        approved_triples = non_conflicting_triples
        if self.enable_verification:
            print("✅ Step 8: Verification")
            print("-" * 70)
            approved_triples = self.verifier_agent.run(
                triples=non_conflicting_triples,
                source_text=raw_text,
            )
            print(f"✓ {len(approved_triples)} triples approved\n")
        else:
            print("⏭️  Step 8: Verification (skipped)\n")

        # Step 9: Integration
        print("💾 Step 9: Knowledge Graph Integration")
        print("-" * 70)
        for triple in approved_triples:
            self.knowledge_graph.add_triple(
                subject=triple.subject,
                relation=triple.relation,
                obj=triple.object,
                confidence=triple.confidence,
                source=triple.source,
                metadata=triple.metadata,
            )
        print(f"✓ Added {len(approved_triples)} triples to knowledge graph\n")

        # Print final stats
        self._print_final_stats()

        return self.knowledge_graph

    def _print_entities_preview(self, entities: list, max_show: int = 5) -> None:
        """Print a preview of entities."""
        for i, entity in enumerate(entities[:max_show]):
            type_str = f" ({entity.type})" if entity.type else ""
            print(f"  • {entity.id}{type_str}")
        if len(entities) > max_show:
            print(f"  ... and {len(entities) - max_show} more")

    def _print_triples_preview(self, triples: list, max_show: int = 5) -> None:
        """Print a preview of triples."""
        for i, triple in enumerate(triples[:max_show]):
            conf_str = f" ({triple.confidence:.2f})" if triple.confidence else ""
            print(f"  • {triple}{conf_str}")
        if len(triples) > max_show:
            print(f"  ... and {len(triples) - max_show} more")

    def _print_final_stats(self) -> None:
        """Print final knowledge graph statistics."""
        print("=" * 70)
        print("FINAL KNOWLEDGE GRAPH STATISTICS")
        print("=" * 70)

        stats = self.knowledge_graph.get_stats()
        print(f"Total Entities: {stats['num_entities']}")
        print(f"Total Triples: {stats['num_triples']}")

        if stats['entity_type_counts']:
            print("\nEntity Types:")
            for etype, count in sorted(
                stats['entity_type_counts'].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  • {etype}: {count}")

        if stats['relation_counts']:
            print("\nRelation Types:")
            for rel, count in sorted(
                stats['relation_counts'].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  • {rel}: {count}")

        print("=" * 70 + "\n")

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Get the current knowledge graph."""
        return self.knowledge_graph

    def export_to_json(self, file_path: str) -> None:
        """
        Export knowledge graph to JSON file.

        Args:
            file_path: Path to output JSON file
        """
        import json

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.knowledge_graph.to_json())

        print(f"✓ Knowledge graph exported to {file_path}")
