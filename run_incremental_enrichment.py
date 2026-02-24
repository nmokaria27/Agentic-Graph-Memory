"""
Demo: Incremental KG Enrichment

Load the existing KG from a previous run, then add new documents
and merge the results. Demonstrates:
- Loading a saved KG
- Running extraction on new documents
- Computing diffs and resolving conflicts
- Merging into the base KG
"""

import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_BACKEND", "ollama").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Add it to .env file.")

from multi_agent_kg.core import (
    LLMConfig,
    IncrementalEnricher,
)
from multi_agent_kg.core.kg_operations import save_kg


def regenerate_visualizations(kg, output_prefix: str = "kg_enriched") -> None:
    """Regenerate PNG and interactive HTML visualizations from a KG."""
    from multi_agent_kg.utils.kg_visualizer import KGVisualizer

    # Build the dict format the visualizer expects
    kg_data = {
        "entities": [
            {"id": e.id, "labels": e.labels, "type": e.type}
            for e in kg.entities.values()
        ],
        "triples": [
            {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "confidence": t.confidence,
            }
            for t in kg.triples
        ],
    }

    viz = KGVisualizer(kg_data=kg_data)

    # Static PNG
    png_path = f"{output_prefix}_static.png"
    result = viz.visualize_static(output_file=png_path)
    if result:
        print(f"  Static PNG saved to: {result}")

    # Interactive HTML
    html_path = f"{output_prefix}_interactive.html"
    result = viz.visualize_interactive(output_file=html_path)
    if result:
        print(f"  Interactive HTML saved to: {result}")


def main():
    # ── 1. Load the existing KG ──────────────────────────────────────────
    kg_path = "kg_export.json"
    if not os.path.exists(kg_path):
        raise SystemExit(
            f"ERROR: {kg_path} not found. Run run_pipeline_on_text.py first "
            "to create the initial KG."
        )

    print("=" * 70)
    print("INCREMENTAL ENRICHMENT DEMO")
    print("=" * 70)

    llm_config = LLMConfig(
        model="gemma3:27b",
        temperature=0.2,
        max_tokens=4096,
    )

    enricher = IncrementalEnricher.from_file(
        kg_path=kg_path,
        llm_config=llm_config,
        match_threshold=0.80,
        auto_resolve_conflicts=True,
    )

    base_stats = enricher.base_kg.get_stats()
    print(f"\nBase KG loaded:")
    print(f"  Entities: {base_stats['num_entities']}")
    print(f"  Triples:  {base_stats['num_triples']}")

    # ── 2. Prepare new documents ─────────────────────────────────────────
    # Example: a follow-up study or supplementary data
    new_documents = [
        {
            "id": "followup_study_01",
            "text": """
A 24-month follow-up of the original cardio-renal-immune cohort (n=380 
of the original 412 participants) confirmed and extended the baseline 
findings. Participants in the highest quartile of the Inflammatory 
Endothelial Stress Score (IESS) at baseline experienced significantly 
greater progression of albuminuria (median uACR increase: +22 mg/g vs 
+3 mg/g in the lowest quartile, p<0.001) and greater increases in LV 
mass index (+8.2 g/m² vs +2.1 g/m², p=0.003).

New biomarker analysis revealed that galectin-3 and sST2 (soluble 
suppression of tumorigenicity 2) were independently associated with 
LV remodeling progression, even after adjusting for IESS and traditional 
risk factors. Galectin-3 levels correlated with fibrosis markers 
(procollagen type III N-terminal propeptide, PIIINP) and predicted 
incident heart failure hospitalization.

Among SGLT2 inhibitor users who continued therapy for 24 months, the 
initial eGFR dip stabilized and reversed, with net eGFR preservation 
of +2.1 mL/min/1.73m² compared to non-users. Empagliflozin specifically 
showed reduction in sC5b-9 levels at 12 months, suggesting direct 
complement-modulating effects beyond glycemic control.

A new sub-analysis of finerenone users (n=34) demonstrated significant 
reductions in uACR (-35% from baseline) and angiopoietin-2 levels, 
with acceptable hyperkalemia rates (serum K >5.5 in 8.8% of patients).

Transcriptomic profiling at 24 months identified a new gene module 
associated with vascular repair (VEGFA, ANGPT1, KDR) that was 
upregulated in participants showing CFR improvement. This repair 
module was inversely correlated with the monocyte activation module 
(S100A8/S100A9), suggesting a balance between inflammatory and 
reparative endothelial programs.
""",
            "metadata": {
                "source": "follow-up study",
                "type": "research_article",
            },
        }
    ]

    # ── 3. Run incremental enrichment ────────────────────────────────────
    report = enricher.add_documents(
        documents=new_documents,
        quality_threshold=0.6,
    )

    # ── 4. Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ENRICHMENT REPORT")
    print("=" * 70)
    print(f"  Documents processed: {report['documents']}")
    print(f"  Elapsed: {report.get('elapsed_seconds', 0):.1f}s")
    print(f"\n  Merge stats:")
    for k, v in report.get("merge_stats", {}).items():
        print(f"    {k}: {v}")
    print(f"  Conflicts resolved: {report.get('conflicts_resolved', 0)}")

    updated_stats = enricher.base_kg.get_stats()
    print(f"\n  Updated KG:")
    print(f"    Entities: {base_stats['num_entities']} → {updated_stats['num_entities']}")
    print(f"    Triples:  {base_stats['num_triples']} → {updated_stats['num_triples']}")

    # ── 5. Save the updated KG ───────────────────────────────────────────
    output_path = "kg_enriched.json"
    enricher.save(output_path)
    print(f"\n  Updated KG saved to: {output_path}")

    # ── 6. Regenerate visualizations ─────────────────────────────────────
    print("\n  Regenerating visualizations...")
    regenerate_visualizations(enricher.base_kg, output_prefix="kg_enriched")
    print("=" * 70)


if __name__ == "__main__":
    main()
