"""
Demo: Domain Expert QA System

Given an existing KG, this script:
1. Loads the KG
2. Builds an OrgChart (clusters entities into domains with topic sub-agents)
3. Initializes domain expert agents
4. Runs interactive QA queries

Usage:
    python run_domain_qa.py                           # demo queries + interactive
    python run_domain_qa.py -q "your question here"   # single question
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LLM_BACKEND", "ollama").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Add it to .env file.")

from multi_agent_kg.core import (
    LLMConfig,
    load_kg,
    DomainBuilder,
)
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator


def main():
    # ── 1. Load KG ───────────────────────────────────────────────────────
    kg_path = "kg_export.json"
    if not os.path.exists(kg_path):
        raise SystemExit(f"ERROR: {kg_path} not found.")

    print("=" * 70)
    print("DOMAIN EXPERT QA SYSTEM")
    print("=" * 70)

    llm_config = LLMConfig(
        model="gemma3:27b",
        temperature=0.2,
        max_tokens=4096,
    )

    kg = load_kg(kg_path)
    stats = kg.get_stats()
    print(f"\nKG loaded: {stats['num_entities']} entities, {stats['num_triples']} triples")

    # ── 2. Build domain structure ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Building domain structure...")
    print("-" * 70)

    builder = DomainBuilder(llm_config)
    org_chart = builder.build(kg)

    print(f"\n{org_chart.domain_summary()}")

    # Save org chart for inspection
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
    print("\nOrg chart saved to: org_chart.json")

    # ── 3. Initialize QA Orchestrator ────────────────────────────────────
    qa = AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=kg,
        llm_config=llm_config,
    )

    # ── 4. Run queries ─────────────────────────────────────────────────
    # Support single-question mode via CLI
    cli_q = globals().get("_cli_question")

    if cli_q:
        # Single question mode
        print("\n" + "=" * 70)
        print("SINGLE-QUESTION MODE")
        print("=" * 70)
        result = qa.query(cli_q)
        print(f"\nQ: {cli_q}")
        print(f"A: {result['final_answer']}")
        print(f"\nCoverage: {result['overall_coverage']:.2f}")
        print(f"Confidence: {result['overall_confidence']:.2f}")
        if result.get("gaps"):
            print(f"Gaps: {result['gaps']}")
        with open("qa_results.json", "w") as f:
            json.dump([result], f, indent=2, default=str)
        print("\nQA results saved to: qa_results.json")
        print("\nGoodbye!")
        return

    # Demo queries
    demo_questions = [
        "How does insulin resistance lead to microvascular dysfunction?",
        "What is the role of complement activation in the cardio-renal-immune network?",
        "Which medications showed cardioprotective effects and what were their mechanisms?",
        "What biomarkers are part of the Inflammatory Endothelial Stress Score?",
    ]

    print("\n" + "=" * 70)
    print("RUNNING DEMO QUERIES")
    print("=" * 70)

    results = []
    for q in demo_questions:
        result = qa.query(q)
        results.append(result)
        print(f"\nQ: {q}")
        print(f"A: {result['final_answer'][:500]}...")
        print(f"   Coverage: {result['overall_coverage']:.2f}")
        print(f"   Confidence: {result['overall_confidence']:.2f}")
        if result.get("gaps"):
            print(f"   Gaps: {result['gaps']}")
        print()

    # Save results
    with open("qa_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("QA results saved to: qa_results.json")

    # ── 5. Interactive mode ──────────────────────────────────────────────
    import sys

    # Skip interactive mode when stdin is not a TTY (e.g. piped or
    # launched by an automated tool).
    if not sys.stdin.isatty():
        print("\nNon-interactive environment detected — skipping interactive QA.")
        print("Run this script directly in a terminal for interactive mode.")
    else:
        print("\n" + "=" * 70)
        print("INTERACTIVE QA MODE (type 'quit' to exit)")
        print("=" * 70)

        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue

                result = qa.query(question)
                print(f"\nAnswer: {result['final_answer']}")
                print(f"\nCoverage: {result['overall_coverage']:.2f}")
                print(f"Confidence: {result['overall_confidence']:.2f}")
                if result.get("gaps"):
                    print(f"Gaps: {result['gaps']}")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Expert QA over a Knowledge Graph")
    parser.add_argument("-q", "--question", type=str, default=None,
                        help="Ask a single question and exit")
    args = parser.parse_args()

    # If a single question is supplied, inject it so main() can pick it up
    _cli_question = args.question
    main()
