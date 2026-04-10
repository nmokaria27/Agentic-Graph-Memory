"""
Run the multi-agent pipeline on pre-extracted text.
Assumes a text file exists in the project root (from extract_pdf.py or manually).
"""

import os
import sys
import json
import argparse

# Resolve project root so file paths work from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
from multi_agent_kg.core import (
    DeliberativeOrchestrator,
    GovernedKnowledgeGraph,
    LLMConfig,
    create_qa_system,
    load_governed_kg,
    save_governed_kg,
)
from multi_agent_kg.utils.debug_logger import DebugLogger

# Load environment
load_dotenv()

if os.getenv("LLM_BACKEND", "ollama").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Add it to .env file.")

# Initialize debug logger (clears previous log)
debug_logger = DebugLogger("pipeline_debug.log", verbose=True, clear_log=True)
print("Debug logging enabled - logs will be saved to pipeline_debug.log\n")

def load_documents(input_path: str) -> list[dict]:
    """Load one text file or a directory of text files into corpus documents."""
    if os.path.isdir(input_path):
        documents = []
        for filename in sorted(os.listdir(input_path)):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(input_path, filename)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            documents.append(
                {
                    "id": os.path.splitext(filename)[0],
                    "text": text,
                    "metadata": {"source": input_path, "type": "research_article"},
                }
            )
        if not documents:
            raise SystemExit(f"ERROR: no .txt files found in {input_path}")
        return documents

    if not os.path.exists(input_path):
        raise SystemExit(f"ERROR: {input_path} not found.")

    with open(input_path, "r", encoding="utf-8") as handle:
        full_text = handle.read()
    return [
        {
            "id": os.path.splitext(os.path.basename(input_path))[0],
            "text": full_text,
            "metadata": {"source": "local_text", "type": "research_article"},
        }
    ]


parser = argparse.ArgumentParser(description="Run the governed KG creation pipeline")
parser.add_argument(
    "--input",
    default=os.getenv("PIPELINE_INPUT", "gfy083_full_plaintext.txt"),
    help="Path to a .txt file or directory of .txt files",
)
parser.add_argument(
    "--governance-mode",
    default=os.getenv("PIPELINE_GOVERNANCE_MODE", "permissive"),
    choices=["strict", "permissive", "audit_only"],
)
args = parser.parse_args()

print("=" * 70)
print("LOADING EXTRACTED TEXT")
print("=" * 70)

documents = load_documents(args.input)
total_chars = sum(len(doc["text"]) for doc in documents)
total_words = sum(len(doc["text"].split()) for doc in documents)
print(f"Loaded {len(documents)} document(s), {total_chars} characters (~{total_words} words)")

print("\n" + "=" * 70)
print("RUNNING MULTI-AGENT PIPELINE")
print("=" * 70)

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
    governed_kg=GovernedKnowledgeGraph(governance_mode=args.governance_mode),
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
    with open("kg_export.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=str)

    print("\nFull knowledge graph saved to: kg_export.json")
    save_governed_kg(orchestrator.governed_kg, "governed_kg_export.json")
    print("Governed knowledge graph saved to: governed_kg_export.json")

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

    from multi_agent_kg.core import DomainBuilder

    governed_for_qa = load_governed_kg("governed_kg_export.json")
    kg_for_qa = governed_for_qa.kg
    kg_qa_stats = kg_for_qa.get_stats()
    print(f"\nKG for QA: {kg_qa_stats['num_entities']} entities, {kg_qa_stats['num_triples']} triples")

    if not governed_for_qa.org_chart.domains:
        print("\nBuilding governed org chart...")
        builder = DomainBuilder(llm_config)
        governed_for_qa.bootstrap_domains(builder)
    print(f"\n{governed_for_qa.org_chart.domain_summary()}")

    qa = create_qa_system(
        governed_kg=governed_for_qa,
        llm_config=llm_config,
        advanced=True,
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
