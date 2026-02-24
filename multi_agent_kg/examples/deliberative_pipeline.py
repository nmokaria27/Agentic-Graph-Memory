"""
Deliberative Multi-Agent Knowledge Graph Pipeline Demo.

This example demonstrates the full integrated pipeline with:
- SharedMemory for cross-document context
- MessageBus for inter-agent communication
- Self-consistency for confidence estimation
- Blackboard voting for ambiguous cases
- Iterative refinement with quality thresholds
- Open-world relation discovery

Usage:
    python -m multi_agent_kg.examples.deliberative_pipeline

Requires:
    OPENAI_API_KEY environment variable
"""

import os
from dotenv import load_dotenv

from multi_agent_kg.core import (
    KnowledgeGraph,
    LLMConfig,
    DeliberativeOrchestrator,
)


def main():
    """Run the deliberative multi-agent pipeline demo."""
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key in .env file or environment")
        return
    
    # Sample documents for processing
    documents = [
        {
            "id": "doc_ml_research",
            "text": """
            Transformer architectures have revolutionized natural language processing. 
            The attention mechanism, introduced by Vaswani et al. in 2017, allows models 
            to weigh the importance of different parts of the input sequence.
            
            BERT, developed by Google, uses bidirectional transformers for pre-training.
            It achieved state-of-the-art results on 11 NLP tasks including question answering
            and named entity recognition. The model was trained on BookCorpus and Wikipedia.
            
            GPT-4, created by OpenAI, is a large language model that demonstrates emergent
            capabilities in reasoning, coding, and creative tasks. It builds upon the 
            transformer architecture with significant scaling improvements.
            
            Recent work by researchers at Stanford, including their Alpaca project, has shown
            that smaller models can be fine-tuned to match larger model performance using
            instruction tuning techniques. This has implications for democratizing AI access.
            """,
            "metadata": {"type": "research", "domain": "machine_learning"},
        },
        {
            "id": "doc_company_news",
            "text": """
            Microsoft announced today that it has completed its acquisition of Activision
            Blizzard for $68.7 billion, making it the largest gaming acquisition in history.
            The deal, first proposed in January 2022, faced regulatory scrutiny in multiple
            jurisdictions before receiving final approval.
            
            Satya Nadella, CEO of Microsoft, stated that this acquisition positions the
            company as a leader in the gaming industry. Bobby Kotick, former CEO of
            Activision Blizzard, will transition out of the company over the coming months.
            
            The combined entity will now own popular franchises including Call of Duty,
            World of Warcraft, Candy Crush, and Overwatch. Microsoft plans to bring these
            titles to its Xbox Game Pass subscription service.
            
            Sony, a major competitor, initially opposed the acquisition citing concerns
            about potential exclusive deals for Call of Duty. Microsoft has committed to
            keeping the franchise available on PlayStation for at least 10 years.
            """,
            "metadata": {"type": "news", "domain": "business"},
        },
    ]
    
    # Configure the orchestrator
    llm_config = LLMConfig(
        model="gemma3:27b",
        temperature=0.3,
        max_tokens=2000,
    )
    
    # Create orchestrator with all features enabled
    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        knowledge_graph=KnowledgeGraph(),
        quality_threshold=0.75,  # Lower for demo
        max_refinement_iterations=2,  # Fewer iterations for demo
        enable_self_consistency=True,
        enable_open_world=True,
        enable_cross_document=True,
    )
    
    # Process the corpus
    print("\n" + "=" * 70)
    print("DELIBERATIVE MULTI-AGENT KNOWLEDGE GRAPH DEMO")
    print("=" * 70)
    print("\nThis demo shows the integrated pipeline with:")
    print("  - SharedMemory for cross-document context")
    print("  - MessageBus for inter-agent communication")
    print("  - Self-consistency for confidence estimation")
    print("  - Iterative refinement with quality thresholds")
    print("  - Open-world relation discovery")
    print("\n" + "=" * 70)
    
    # Process all documents
    results = orchestrator.process_corpus(documents)
    
    # Show final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    stats = orchestrator.get_stats()
    
    print(f"\nDocuments Processed: {stats['documents_processed']}")
    print(f"\nKnowledge Graph:")
    kg_stats = stats['kg_stats']
    print(f"  Entities: {kg_stats.get('total_entities', 0)}")
    print(f"  Triples: {kg_stats.get('total_triples', 0)}")
    print(f"  Unique Relations: {kg_stats.get('unique_relations', 0)}")
    
    print(f"\nMemory System:")
    mem_stats = stats['memory_stats']
    print(f"  Total Memories: {mem_stats.get('total_memories', 0)}")
    print(f"  Blackboard Entries: {mem_stats.get('blackboard_entries', 0)}")
    print(f"  Entity Aliases: {mem_stats.get('entity_aliases', 0)}")
    
    print(f"\nDiscovered Relations:")
    for rel in stats.get('discovered_relations', [])[:10]:
        print(f"  - {rel}")
    
    # Export knowledge graph
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH EXPORT (sample)")
    print("=" * 70)
    
    export = orchestrator.export()
    kg = export['knowledge_graph']
    
    print("\nEntities (first 10):")
    for entity in kg['entities'][:10]:
        print(f"  {entity['id']}: {entity['type']}")
    
    print("\nTriples (first 10):")
    for triple in kg['triples'][:10]:
        print(f"  ({triple['subject']}) --[{triple['relation']}]--> ({triple['object']})")
        print(f"    Confidence: {triple.get('confidence', 0):.2f}")
    
    # Visualize the knowledge graph
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 70)
    
    try:
        # Generate interactive HTML visualization
        viz_file = orchestrator.visualize_kg(
            output_file="kg_interactive.html",
            layout="spring",
            show_labels=True,
            generate_static=True  # Also generate PNG
        )
        print(f"\n✓ Visualization saved to: {viz_file}")
        print(f"  Open in browser to explore interactively!")
    except ImportError as e:
        print(f"\n⚠ Visualization skipped: {e}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

