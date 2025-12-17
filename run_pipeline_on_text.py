"""
Run the multi-agent pipeline on pre-extracted text.
Assumes article_text.txt exists from running extract_pdf.py
"""

from dotenv import load_dotenv
import os
from multi_agent_kg.core import LLMConfig, DeliberativeOrchestrator, KnowledgeGraph

# Load environment
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Add it to .env file.")

# Load the extracted text
text_file = "article_text_short.txt"
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

# Configure LLM (using gpt-4o-mini for cost efficiency)
llm_config = LLMConfig(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=4096,  # Model maximum for gpt-4o-mini
)

# Create orchestrator
print("\nInitializing orchestrator...")
orchestrator = DeliberativeOrchestrator(
    llm_config=llm_config,
    knowledge_graph=KnowledgeGraph(),
    quality_threshold=0.7,
    max_refinement_iterations=1,
    enable_self_consistency=False,
    enable_open_world=True,
    enable_cross_document=False,
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
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nFull knowledge graph saved to: kg_export.json")

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)
    print(f"\n{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
