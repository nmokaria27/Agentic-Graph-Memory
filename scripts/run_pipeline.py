"""
Run the multi-agent pipeline on pre-extracted text.
Assumes a text file exists in the project root (from extract_pdf.py or manually).
"""

import os
import sys

# Resolve project root so file paths work from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
from multi_agent_kg.core import LLMConfig, DeliberativeOrchestrator, KnowledgeGraph
from multi_agent_kg.utils.debug_logger import DebugLogger

# Load environment
load_dotenv()

if os.getenv("LLM_BACKEND", "ollama").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Add it to .env file.")

# Initialize debug logger (clears previous log)
debug_logger = DebugLogger("pipeline_debug.log", verbose=True, clear_log=True)
print("Debug logging enabled - logs will be saved to pipeline_debug.log\n")

# Load the extracted text
text_file = "gfy083_full_plaintext.txt"
if not os.path.exists(text_file):
    raise SystemExit(f"ERROR: {text_file} not found. Run extract_pdf.py first.")

print("=" * 70)
print("LOADING EXTRACTED TEXT")
print("=" * 70)

with open(text_file, "r", encoding="utf-8") as f:
    full_text = f.read()

print(f"Loaded {len(full_text)} characters (~{len(full_text.split())} words)")

print("\n" + "=" * 70)
print("RUNNING MULTI-AGENT PIPELINE")
print("=" * 70)

# Prepare document
documents = [
    {
        "id": "pubmed_article_lancet",
        "text": full_text,
        "metadata": {
            "source": "pubmed",
            "type": "research_article",
        }
    }
]

# Configure LLM (using gemma3:27b via Ollama for best quality)
llm_config = LLMConfig(
    model="gemma3:27b",
    temperature=0.2,
    max_tokens=4096,
)

# Create orchestrator
print("\nInitializing orchestrator...")
orchestrator = DeliberativeOrchestrator(
    llm_config=llm_config,
    knowledge_graph=KnowledgeGraph(),
    quality_threshold=0.5,  # Lowered for maximum recall; garbage filtered by verification
    max_refinement_iterations=1,
    enable_self_consistency=False,
    enable_open_world=True,
    enable_cross_document=False,
    debug_logger=debug_logger,
)

print("Processing document through pipeline...")
print("This may take several minutes...\n")

# Process the document
try:
    results = orchestrator.process_corpus(documents)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Get statistics
    stats = orchestrator.get_stats()
    
    print(f"\nDocuments Processed: {stats['documents_processed']}")
    
    print(f"\nKnowledge Graph:")
    kg_stats = stats['kg_stats']
    print(f"  Total Entities: {kg_stats.get('total_entities', 0)}")
    print(f"  Total Triples: {kg_stats.get('total_triples', 0)}")
    print(f"  Unique Relations: {kg_stats.get('unique_relations', 0)}")
    
    print(f"\nMemory System:")
    mem_stats = stats['memory_stats']
    print(f"  Total Memories: {mem_stats.get('total_memories', 0)}")
    print(f"  Blackboard Entries: {mem_stats.get('blackboard_entries', 0)}")
    print(f"  Entity Aliases: {mem_stats.get('entity_aliases', 0)}")
    
    # Export knowledge graph
    export = orchestrator.export()
    kg = export['knowledge_graph']
    
    print("\n" + "=" * 70)
    print("EXTRACTED ENTITIES")
    print("=" * 70)
    
    if kg['entities']:
        print(f"\nShowing first 20 of {len(kg['entities'])} entities:")
        for entity in kg['entities'][:20]:
            print(f"  - {entity['id']} ({entity['type']})")
    else:
        print("\nNo entities extracted.")
    
    print("\n" + "=" * 70)
    print("EXTRACTED TRIPLES")
    print("=" * 70)
    
    if kg['triples']:
        print(f"\nShowing first 20 of {len(kg['triples'])} triples:")
        for triple in kg['triples'][:20]:
            print(f"  ({triple['subject']}) --[{triple['relation']}]--> ({triple['object']})")
            if 'confidence' in triple:
                print(f"    Confidence: {triple['confidence']:.2f}")
    else:
        print("\nNo triples extracted.")
    
    # Save full export to file
    import json
    with open("kg_export.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=str)

    print("\nFull knowledge graph saved to: kg_export.json")

    # Invalidate org chart cache since KG changed
    if os.path.exists("org_chart_cache.json"):
        os.remove("org_chart_cache.json")
        print("  (org_chart_cache.json removed — will rebuild on next qa_server start)")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Import visualizer
    from multi_agent_kg.utils.kg_visualizer import KGVisualizer
    
    # Create visualizer with the exported KG data
    kg_data = export.get('knowledge_graph', export)
    visualizer = KGVisualizer(kg_data=kg_data)
    
    viz_html = "kg_interactive.html"
    viz_png = "kg_static.png"
    
    # Interactive HTML visualization
    visualizer.visualize_interactive(
        output_file=viz_html,
        height="800px"
    )
    print(f"✓ Interactive visualization: {viz_html}")

    # Static PNG visualization
    visualizer.visualize_static(
        output_file=viz_png,
        layout="spring",
        figsize=(20, 16)
    )
    print(f"✓ Static visualization: {viz_png}")

    # ═══════════════════════════════════════════════════════════════
    # QA DEMO
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("QA DEMO")
    print("=" * 70)

    from multi_agent_kg.core import load_kg, DomainBuilder
    from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator

    # Build domain structure from KG
    kg_for_qa = load_kg("kg_export.json")
    kg_qa_stats = kg_for_qa.get_stats()
    print(f"\nKG for QA: {kg_qa_stats['num_entities']} entities, {kg_qa_stats['num_triples']} triples")

    print("\nBuilding domain structure...")
    builder = DomainBuilder(llm_config)
    org_chart = builder.build(kg_for_qa)
    print(f"\n{org_chart.domain_summary()}")

    qa = AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=kg_for_qa,
        llm_config=llm_config,
    )

    test_questions = [
        "How does insulin resistance lead to microvascular dysfunction?",
        "What biomarkers are associated with endothelial dysfunction?",
        "What is the relationship between IL-6 signaling and cardiovascular outcomes?",
        "What role do SGLT2 inhibitors play in cardiovascular protection?",
    ]

    qa_results_list = []
    for q in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}")
        result = qa.query(q)
        qa_results_list.append(result)
        print(f"\nA: {result['final_answer'][:600]}")
        print(f"Coverage: {result['overall_coverage']:.2f}  Confidence: {result['overall_confidence']:.2f}")

    with open("qa_results.json", "w", encoding="utf-8") as f:
        json.dump(qa_results_list, f, indent=2, default=str)
    print(f"\nQA results saved to: qa_results.json")

    # ═══════════════════════════════════════════════════════════════
    # ENHANCED VISUALIZER WITH QA
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GENERATING ENHANCED EXPLORER")
    print("=" * 70)

    explorer_html = "kg_explorer.html"
    visualizer.visualize_interactive_enhanced(
        output_file=explorer_html,
        qa_results=qa_results_list,
    )
    print(f"✓ Enhanced explorer: {explorer_html}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOpen {explorer_html} in your browser to explore the knowledge graph with QA!")

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)
    print(f"\n{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
