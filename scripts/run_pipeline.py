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
    CheckpointManager,
    DeliberativeOrchestrator,
    GovernedKnowledgeGraph,
    LLMConfig,
    create_qa_system,
    discover_checkpoints,
    load_governed_kg,
    save_governed_kg,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
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
    default=os.getenv("PIPELINE_GOVERNANCE_MODE", "audit_only"),
    choices=["strict", "permissive", "audit_only"],
)
parser.add_argument(
    "--model",
    default=os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b"),
    help="Ollama model to use for all pipeline agents (default: gemma4:31b)",
)
parser.add_argument(
    "--resume",
    "-r",
    action="store_true",
    help="Resume from per-stage checkpoints under --checkpoint-dir (skips completed stages).",
)
parser.add_argument(
    "--checkpoint-dir",
    default="checkpoints",
    help="Directory to store per-document, per-stage checkpoints (default: checkpoints/).",
)
parser.add_argument(
    "--no-checkpoint",
    action="store_true",
    help="Disable checkpointing entirely (overrides --checkpoint-dir).",
)
parser.add_argument(
    "--skip-relink",
    action="store_true",
    help="Skip the orphan relink pass and value-orphan attribute folding.",
)
args = parser.parse_args()
CHECKPOINT_DIR = None if args.no_checkpoint else args.checkpoint_dir

print("=" * 70)
print("LOADING EXTRACTED TEXT")
print("=" * 70)

documents = load_documents(args.input)

# Check for resume — per-document, per-stage checkpoint discovery.
initial_kg = None
initial_governed_kg = None

if args.resume and CHECKPOINT_DIR is not None:
    existing = discover_checkpoints(CHECKPOINT_DIR)
    relevant = {entry["document_id"]: entry for entry in existing}
    resumable = [doc for doc in documents if doc["id"] in relevant]
    if not resumable:
        print(f"--resume set but no checkpoints found under {CHECKPOINT_DIR}/. Starting from scratch.")
    else:
        print(f"--resume: found checkpoints for {len(resumable)} doc(s):")
        for doc in resumable:
            entry = relevant[doc["id"]]
            stages = entry.get("completed_stages", [])
            print(f"  - {doc['id']}: last_stage={entry.get('last_stage')}, completed={stages}")

        # Pick the most-advanced doc's KG snapshot as the orchestrator's starting
        # governance state. Per-doc checkpoint files restore per-stage progress.
        latest = max(
            resumable,
            key=lambda doc: len(relevant[doc["id"]].get("completed_stages", [])),
        )
        latest_dir = os.path.join(CHECKPOINT_DIR, latest["id"])
        gov_path = os.path.join(latest_dir, "governed_kg_latest.json")
        kg_path = os.path.join(latest_dir, "knowledge_graph_latest.json")
        if os.path.exists(gov_path):
            try:
                initial_governed_kg = load_governed_kg(gov_path)
                initial_kg = initial_governed_kg.kg
                print(f"Restored governed_kg snapshot from {gov_path}")
            except Exception as e:
                print(f"WARNING: could not load governed_kg snapshot at {gov_path}: {e}")
        elif os.path.exists(kg_path):
            try:
                with open(kg_path, "r", encoding="utf-8") as f:
                    initial_kg = KnowledgeGraph.from_dict(json.load(f))
                print(f"Restored knowledge_graph snapshot from {kg_path}")
            except Exception as e:
                print(f"WARNING: could not load knowledge_graph snapshot at {kg_path}: {e}")
elif args.resume and CHECKPOINT_DIR is None:
    print("WARNING: --resume requires checkpointing; --no-checkpoint cancels it. Starting from scratch.")
elif not args.resume and CHECKPOINT_DIR is not None:
    # Fresh run with existing checkpoints is destructive: the new run will
    # overwrite stage files + governed_kg_latest.json as it progresses,
    # leaving stale files mixed with new ones if it crashes early. Refuse
    # unless the user explicitly opts in.
    existing = discover_checkpoints(CHECKPOINT_DIR)
    collision = [entry for entry in existing if entry["document_id"] in {d["id"] for d in documents}]
    if collision:
        print("=" * 70)
        print("ERROR: existing checkpoints would be overwritten by this fresh run:")
        for entry in collision:
            print(f"  - {entry['document_id']}: completed={entry.get('completed_stages', [])}")
        print()
        print("Options:")
        print("  1. python3 scripts/run_pipeline.py --input ... --resume    (continue from checkpoint)")
        print("  2. rm -rf checkpoints/<document_id>                        (delete then re-run fresh)")
        print("  3. python3 scripts/run_pipeline.py --input ... --checkpoint-dir checkpoints_new/")
        print("=" * 70)
        raise SystemExit(1)

total_chars = sum(len(doc["text"]) for doc in documents)
total_words = sum(len(doc["text"].split()) for doc in documents)
print(f"Loaded {len(documents)} document(s), {total_chars} characters (~{total_words} words)")

print("\n" + "=" * 70)
print("RUNNING MULTI-AGENT PIPELINE")
print("=" * 70)

# Configure LLM. --model only seeds LLM_DEFAULT_MODEL; per-tier env vars
# (LLM_SMALL/MEDIUM/LARGE_MODEL) win over defaults, which now come from
# multi_agent_kg.agents.base.get_default_model_tiers (qwen3:8b / gemma3:27b
# / gemma4:31b per HANDOFF.md §4).
os.environ.setdefault("LLM_DEFAULT_MODEL", args.model)
print(f"Default model (LLM_DEFAULT_MODEL): {args.model}")
print(
    "Tier resolution: "
    f"SMALL={os.getenv('LLM_SMALL_MODEL', 'qwen3:8b')}, "
    f"MEDIUM={os.getenv('LLM_MEDIUM_MODEL', 'gemma3:27b')}, "
    f"LARGE={os.getenv('LLM_LARGE_MODEL', 'gemma4:31b')}"
)

llm_config = LLMConfig(
    model=args.model,
    temperature=0.2,
    max_tokens=4096,
)

# Create orchestrator
print("\nInitializing orchestrator...")
orchestrator = DeliberativeOrchestrator(
    llm_config=llm_config,
    knowledge_graph=initial_kg,
    governed_kg=initial_governed_kg or GovernedKnowledgeGraph(governance_mode=args.governance_mode),
    quality_threshold=0.35,  # Lowered for tuning Stage 1
    max_refinement_iterations=1,
    enable_self_consistency=False,
    enable_open_world=True,
    enable_cross_document=False,
    debug_logger=debug_logger,
    checkpoint_dir=CHECKPOINT_DIR,
    resume=args.resume,
)
if args.governance_mode == "strict":
    print("Strict governance enabled: creation will request explicit review before committing triples.")
elif args.governance_mode == "permissive":
    print("Permissive governance enabled: triples are routed and audited, then accepted.")
else:
    print("Audit-only governance enabled: creation preserves an audit trail with no review latency.")

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

    # ── Orphan relink pass + value-orphan folding (B3/B4) ────────────
    if not args.skip_relink:
        from multi_agent_kg.core.config import RetrievalConfig
        from multi_agent_kg.core.orphan_relink import fold_value_orphans, relink_orphans
        from multi_agent_kg.core.vector_index import KGVectorStore

        retrieval_config = RetrievalConfig()
        vector_store = None
        VECTOR_CACHE_DIR = os.getenv("VECTOR_CACHE_DIR", "governed_kg_export.vectors")
        if retrieval_config.use_vectors:
            try:
                from multi_agent_kg.core.vector_index import KGVectorStore
                # Try loading cached index first — saves all embedding calls when
                # the KG content hasn't changed (hash-verified).
                vector_store = KGVectorStore.load_dir(
                    VECTOR_CACHE_DIR,
                    orchestrator.governed_kg,
                    model=retrieval_config.embedding_model,
                )
                if vector_store is not None:
                    print(f"  Loaded vector index from cache: {VECTOR_CACHE_DIR}")
                else:
                    vector_store = KGVectorStore(model=retrieval_config.embedding_model)
                    vector_store.build(orchestrator.governed_kg)
                    vector_store.save_dir(VECTOR_CACHE_DIR)
                    print(f"  Built and cached vector index: {VECTOR_CACHE_DIR}")
            except Exception as exc:
                print(f"WARNING: vector store unavailable for relink ({exc})")
                vector_store = None

        print("\n" + "=" * 70)
        print("ORPHAN RELINK PASS")
        print("=" * 70)
        relink_stats = relink_orphans(
            orchestrator.governed_kg,
            llm_config=llm_config,
            retrieval_config=retrieval_config,
            vector_store=vector_store,
        )
        print(f"  Orphans before: {relink_stats['orphans_before']}")
        print(f"  LLM calls: {relink_stats['llm_calls']}")
        print(f"  Triples proposed: {relink_stats['triples_proposed']}")
        print(f"  Triples committed: {relink_stats['triples_committed']}")
        print(f"  Orphans after relink: {relink_stats['orphans_after']}")

        fold_stats = fold_value_orphans(orchestrator.governed_kg)
        print(f"  Value-orphans folded into hosts: {fold_stats['folded']} "
              f"(no host found: {fold_stats['no_host']})")

        # Relink + fold commit new triples and may drop value-orphans, so the
        # cached index saved earlier no longer matches the KG content hash.
        # Refresh and re-save so the next run/QA server loads from cache instead
        # of re-embedding the whole graph.
        if vector_store is not None and (
            relink_stats.get("triples_committed") or fold_stats.get("folded")
        ):
            try:
                vector_store.build(orchestrator.governed_kg)
                vector_store.save_dir(VECTOR_CACHE_DIR)
                print(f"  Refreshed vector index cache after relink/fold: {VECTOR_CACHE_DIR}")
            except Exception as exc:
                print(f"WARNING: failed to refresh vector index cache ({exc})")

    print(f"\nGoverned KG:")
    governed_stats = orchestrator.governed_kg.get_stats()
    print(f"  Orphan Entities: {governed_stats.get('orphan_entities', 0)} "
          f"of {governed_stats.get('entities', 0)}")
    print(f"  Domains: {governed_stats.get('domains', 0)}")
    print(f"  Cross-domain Relations: {governed_stats.get('cross_domain_relations', 0)}")
    print(f"  Assignment Counts: {governed_stats.get('assignment_counts', {})}")
    print(f"  Bootstrap Assignment Stats: {governed_stats.get('bootstrap_assignment_stats', {})}")
    
    print(f"\nMemory System:")
    mem_stats = stats['memory_stats']
    print(f"  Total Memories: {mem_stats.get('total_memories', 0)}")
    print(f"  Blackboard Entries: {mem_stats.get('blackboard_entries', 0)}")
    print(f"  Entity Aliases: {mem_stats.get('entity_aliases', 0)}")
    
    # Export knowledge graph
    export = orchestrator.export()
    kg = export['knowledge_graph']
    
    # Calculate Connectivity Metrics
    try:
        import networkx as nx
        G = nx.Graph()
        if kg.get('entities'):
            G.add_nodes_from([e['id'] for e in kg.get('entities', []) if 'id' in e])
        if kg.get('triples'):
            for t in kg.get('triples', []):
                subj = t.get('subject_id') or t.get('subject')
                obj = t.get('object_id') or t.get('object')
                if subj and obj:
                    G.add_edge(subj, obj)
        
        components = list(nx.connected_components(G))
        if components:
            largest_component = max(components, key=len)
            largest_component_size = len(largest_component)
            orphan_count = sum(1 for c in components if len(c) == 1)
            num_entities = len(G.nodes)
            connectivity_pct = (largest_component_size / num_entities * 100) if num_entities > 0 else 0
            
            print(f"\nConnectivity Metrics:")
            print(f"  Largest Component Size: {largest_component_size} / {num_entities} ({connectivity_pct:.1f}%)")
            print(f"  Orphan Count: {orphan_count}")
    except ImportError:
        print("\nConnectivity Metrics: networkx not installed, skipping.")

    
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

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nUse scripts/run_demo.py or scripts/qa_server.py if you want to test QA on top of this governed graph.")

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)
    print(f"\n{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
