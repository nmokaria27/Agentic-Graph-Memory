"""
Full pipeline test: Incremental Enrichment + Domain QA

1. Loads existing KG from kg_export.json
2. Adds a new document (SGLT2 inhibitor mechanistic follow-up study)
3. Saves enriched KG
4. Builds OrgChart and runs QA queries
5. Saves all results (kg_enriched.json, org_chart.json, qa_results.json)
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from multi_agent_kg.core import (
    LLMConfig,
    IncrementalEnricher,
    load_kg,
    DomainBuilder,
)
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator

def main():
    llm_config = LLMConfig(
        model="gemma3:27b",
        temperature=0.2,
        max_tokens=4096,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: INCREMENTAL ENRICHMENT
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: INCREMENTAL ENRICHMENT")
    print("=" * 70)

    kg_path = "kg_export.json"
    if not os.path.exists(kg_path):
        raise SystemExit(f"ERROR: {kg_path} not found. Run run_pipeline_on_text.py first.")

    enricher = IncrementalEnricher.from_file(
        kg_path=kg_path,
        llm_config=llm_config,
        match_threshold=0.80,
        auto_resolve_conflicts=True,
    )

    base_stats = enricher.base_kg.get_stats()
    print(f"\nBase KG: {base_stats['num_entities']} entities, {base_stats['num_triples']} triples")

    # Load the new study text
    with open("new_study_text.txt", "r") as f:
        new_text = f.read()

    new_documents = [
        {
            "id": "sglt2_complement_study",
            "text": new_text,
            "metadata": {
                "source": "mechanistic follow-up study",
                "type": "research_article",
            },
        }
    ]

    report = enricher.add_documents(documents=new_documents, quality_threshold=0.5)

    print("\n" + "=" * 70)
    print("ENRICHMENT REPORT")
    print("=" * 70)
    print(f"  Documents processed: {report['documents']}")
    print(f"  Elapsed: {report.get('elapsed_seconds', 0):.1f}s")
    for k, v in report.get("merge_stats", {}).items():
        print(f"  {k}: {v}")

    updated_stats = enricher.base_kg.get_stats()
    print(f"\n  KG: {base_stats['num_entities']} → {updated_stats['num_entities']} entities")
    print(f"  KG: {base_stats['num_triples']} → {updated_stats['num_triples']} triples")

    enricher.save("kg_enriched.json")
    print(f"\n  Enriched KG saved to: kg_enriched.json")

    # Invalidate org chart cache since KG changed
    if os.path.exists("org_chart_cache.json"):
        os.remove("org_chart_cache.json")
        print("  (org_chart_cache.json removed — will rebuild on next qa_server start)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: DOMAIN QA
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: DOMAIN QA SYSTEM")
    print("=" * 70)

    # Use the enriched KG for QA
    kg = enricher.base_kg
    stats = kg.get_stats()
    print(f"\nKG for QA: {stats['num_entities']} entities, {stats['num_triples']} triples")

    # Build domain structure
    print("\nBuilding domain structure...")
    builder = DomainBuilder(llm_config)
    org_chart = builder.build(kg)
    print(f"\n{org_chart.domain_summary()}")

    # Save org chart
    org_data = {
        "domains": [
            {
                "domain_id": d.domain_id,
                "label": d.label,
                "description": d.description,
                "entity_count": len(d.entity_ids),
                "entities": list(d.entity_ids)[:20],
                "relation_schema": d.relation_schema,
                "topics": [
                    {
                        "topic_id": t.topic_id,
                        "label": t.label,
                        "description": t.description,
                        "entity_count": len(t.entity_ids),
                        "keywords": t.keywords,
                    }
                    for t in d.topics
                ],
            }
            for d in org_chart.domains
        ],
        "cross_domain_relations": len(org_chart.cross_domain_relations),
    }
    with open("org_chart.json", "w") as f:
        json.dump(org_data, f, indent=2)
    print("Org chart saved to: org_chart.json")

    # Initialize QA
    qa = AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=kg,
        llm_config=llm_config,
    )

    # Test questions that span both documents
    test_questions = [
        # From original document
        "How does insulin resistance lead to microvascular dysfunction?",
        # From new document
        "How does empagliflozin modulate complement activation?",
        # Cross-document reasoning
        "What is the relationship between IL-6 signaling and complement activation, and how do SGLT2 inhibitors affect this pathway?",
        # Specific factual question
        "What biomarkers are part of the Inflammatory Endothelial Stress Score (IESS)?",
    ]

    print("\n" + "=" * 70)
    print("RUNNING QA QUERIES")
    print("=" * 70)

    results = []
    for q in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}")

        result = qa.query(q)
        results.append(result)

        print(f"\nA: {result['final_answer'][:800]}")
        print(f"\nCoverage: {result['overall_coverage']:.2f}")
        print(f"Confidence: {result['overall_confidence']:.2f}")
        if result.get("gaps"):
            print(f"Gaps: {result['gaps']}")
        if result.get("domain_responses"):
            print(f"Domains consulted: {len(result['domain_responses'])}")
            for dr in result["domain_responses"]:
                print(f"  - {dr.get('domain_id', '?')}: conf={dr.get('confidence', 0):.2f}")

    # Save QA results
    with open("qa_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nQA results saved to: qa_results.json")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL TEST COMPLETE")
    print("=" * 70)
    print(f"\nFiles generated:")
    print(f"  kg_enriched.json  - Enriched KG with new document")
    print(f"  org_chart.json    - Domain organization structure")
    print(f"  qa_results.json   - QA query results")
    print(f"\nKG Stats:")
    print(f"  Entities: {base_stats['num_entities']} → {updated_stats['num_entities']}")
    print(f"  Triples:  {base_stats['num_triples']} → {updated_stats['num_triples']}")
    print(f"\nQA Summary:")
    for i, r in enumerate(results):
        print(f"  Q{i+1}: coverage={r['overall_coverage']:.2f}, confidence={r['overall_confidence']:.2f}")


if __name__ == "__main__":
    main()
