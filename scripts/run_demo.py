#!/usr/bin/env python3
"""
Demonstration script: Domain Expert QA + Advanced QA + KGAFE Evaluation.

Uses the existing kg_export.json to showcase:
  1. Domain clustering and expert routing (the org chart)
  2. Advanced QA with active exploration, debate, critic, provenance
  3. KGAFE evaluation of QA answers (atomic fact decomposition + 3-tier verification)

This script produces demo_results.json with all outputs for inspection.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from multi_agent_kg.core import (
    DomainBuilder,
    LLMConfig,
    create_qa_system,
    load_governed_kg,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

KG_PATH = "governed_kg_export.json"
OUTPUT_PATH = "demo_results.json"

llm_config = LLMConfig(model="gemma4:31b", temperature=0.2, max_tokens=4096)

# Questions that exercise different QA capabilities
DEMO_QUESTIONS = [
    # Single-domain, multi-hop reasoning
    "How does insulin resistance contribute to endothelial dysfunction?",
    # Cross-domain question requiring multiple experts
    "What is the relationship between IL-6 signaling, complement activation, and cardiovascular outcomes?",
    # Specific factual lookup
    "Which SGLT2 inhibitors are mentioned and what are their cardiovascular effects?",
    # Follow-up question (tests session memory)
    "What biomarkers are associated with the conditions you just described?",
]


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


def main():
    all_results = {}

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Load KG and build domain structure
    # ──────────────────────────────────────────────────────────────────
    section("STEP 1: LOADING KG AND BUILDING DOMAIN STRUCTURE")

    if not os.path.exists(KG_PATH):
        raise SystemExit(f"ERROR: {KG_PATH} not found. Run run_pipeline_on_text.py first.")

    governed_kg = load_governed_kg(KG_PATH)
    kg = governed_kg.kg
    stats = kg.get_stats()
    print(f"KG loaded: {stats['num_entities']} entities, {stats['num_triples']} triples")

    # Show entity type distribution
    type_counts = {}
    for eid, entity in kg.entities.items():
        t = entity.type or "Unknown"
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nEntity type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:35s} {c:3d}")

    # Show relation type distribution
    rel_counts = {}
    for triple in kg.triples:
        r = triple.relation
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print(f"\nRelation type distribution (top 15):")
    for r, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {r:40s} {c:3d}")

    all_results["kg_stats"] = {
        "num_entities": stats["num_entities"],
        "num_triples": stats["num_triples"],
        "entity_types": type_counts,
        "relation_types": rel_counts,
    }

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Build OrgChart (domain clustering)
    # ──────────────────────────────────────────────────────────────────
    section("STEP 2: BUILDING ORG CHART (DOMAIN EXPERT CLUSTERING)")

    t0 = time.time()
    builder = DomainBuilder(llm_config)
    if not governed_kg.org_chart.domains:
        governed_kg.bootstrap_domains(builder)
    org_chart = governed_kg.org_chart
    build_time = time.time() - t0

    print(f"Org chart built in {build_time:.1f}s\n")
    print(org_chart.domain_summary())

    # Detailed domain breakdown
    org_chart_data = []
    for d in org_chart.domains:
        domain_info = {
            "domain_id": d.domain_id,
            "label": d.label,
            "description": d.description,
            "num_entities": len(d.entity_ids),
            "sample_entities": sorted(list(d.entity_ids))[:10],
            "relation_schema": d.relation_schema,
            "topics": [],
        }
        print(f"\n{'─'*60}")
        print(f"DOMAIN: {d.label} ({d.domain_id})")
        print(f"  Description: {d.description}")
        print(f"  Entities: {len(d.entity_ids)}")
        print(f"  Sample entities: {sorted(list(d.entity_ids))[:8]}")
        print(f"  Relation schema: {list(d.relation_schema.keys())[:5]}")

        for t in d.topics:
            topic_info = {
                "topic_id": t.topic_id,
                "label": t.label,
                "description": t.description,
                "num_entities": len(t.entity_ids),
                "keywords": t.keywords,
            }
            domain_info["topics"].append(topic_info)
            print(f"    Topic: {t.label}")
            print(f"      Entities: {len(t.entity_ids)}, Keywords: {t.keywords[:5]}")

        org_chart_data.append(domain_info)

    print(f"\nCross-domain relations: {len(org_chart.cross_domain_relations)}")

    all_results["org_chart"] = {
        "domains": org_chart_data,
        "cross_domain_relations": len(org_chart.cross_domain_relations),
        "build_time_seconds": round(build_time, 2),
    }

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Run Advanced QA queries
    # ──────────────────────────────────────────────────────────────────
    section("STEP 3: ADVANCED QA — ACTIVE EXPLORATION + DEBATE + CRITIC + PROVENANCE")

    qa = create_qa_system(
        governed_kg=governed_kg,
        llm_config=llm_config,
        advanced=True,
        max_exploration_rounds=3,
        enable_debate=True,
        enable_critic=True,
    )

    qa_results = []
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        section(f"QUERY {i}/{len(DEMO_QUESTIONS)}")
        print(f"Q: {question}\n")

        t0 = time.time()
        result = qa.query(question)
        elapsed = time.time() - t0

        # Print detailed breakdown
        print(f"\n{'─'*60}")
        print(f"ANSWER:")
        print(f"{result['final_answer']}")
        print(f"\n{'─'*60}")
        print(f"METRICS:")
        print(f"  Coverage:   {result['overall_coverage']:.2f}")
        print(f"  Confidence: {result['overall_confidence']:.2f}")
        print(f"  Duration:   {elapsed:.1f}s")

        if result.get("gaps"):
            print(f"  Gaps: {result['gaps']}")

        # Sub-question routing
        print(f"\nSUB-QUESTION ROUTING:")
        for sq in result.get("sub_questions", []):
            print(f"  Q: {sq.get('question', '?')}")
            print(f"    -> Domains: {sq.get('target_domains', [])}")

        # Domain expert responses
        print(f"\nDOMAIN EXPERT RESPONSES:")
        for dr in result.get("domain_responses", []):
            rounds = dr.get("exploration_rounds", 1)
            print(f"  [{dr.get('domain_id', '?')}]")
            print(f"    Coverage:  {dr.get('coverage', 0):.2f}")
            print(f"    Confidence: {dr.get('confidence', 0):.2f}")
            print(f"    Exploration rounds: {rounds}")
            if dr.get("exploration_trace"):
                for tr in dr["exploration_trace"]:
                    print(f"      Round {tr['round']}: conf={tr['confidence']:.2f}, "
                          f"entities_explored={tr['entities_explored']}")
            print(f"    Evidence: {dr.get('evidence', [])[:3]}")

        # Debate results
        if result.get("debate_results"):
            print(f"\nDEBATE RESULTS:")
            for db in result["debate_results"]:
                conflict = db.get("conflict", {})
                resolution = db.get("resolution", {})
                print(f"  Conflict: {conflict.get('description', '?')}")
                print(f"  Domain A ({conflict.get('domain_a', '?')}): {conflict.get('claim_a', '?')[:100]}")
                print(f"  Domain B ({conflict.get('domain_b', '?')}): {conflict.get('claim_b', '?')[:100]}")
                print(f"  Resolution: {resolution.get('resolution', '?')[:150]}")
                print(f"  Winner: {resolution.get('winning_domain', '?')}")
        else:
            print(f"\n  No conflicts detected between experts.")

        # Critic results
        critic = result.get("critic_result")
        if critic:
            print(f"\nCRITIC REVIEW:")
            print(f"  Approved: {critic.get('approved', '?')}")
            print(f"  Severity: {critic.get('overall_severity', '?')}")
            issues = critic.get("issues", [])
            if issues:
                print(f"  Issues ({len(issues)}):")
                for issue in issues[:3]:
                    print(f"    [{issue.get('severity', '?')}] {issue.get('type', '?')}: "
                          f"{issue.get('description', '?')[:100]}")
            if critic.get("revised_answer"):
                print(f"  Revised answer: {critic['revised_answer'][:200]}...")
        else:
            print(f"\n  Critic disabled or not run.")

        # Provenance
        prov = result.get("provenance", {})
        print(f"\nPROVENANCE CHAIN:")
        print(f"  Total claims:    {prov.get('total_claims', 0)}")
        print(f"  Grounded claims: {prov.get('grounded_claims', 0)}")
        print(f"  Ungrounded:      {prov.get('ungrounded_claims', 0)}")
        print(f"  Groundedness:    {prov.get('groundedness_ratio', 0):.2%}")
        for rec in prov.get("records", [])[:3]:
            print(f"    Claim: {rec.get('claim', '?')[:80]}")
            print(f"      Type: {rec.get('evidence_type', '?')}, "
                  f"Confidence: {rec.get('confidence', 0):.2f}, "
                  f"Triples: {len(rec.get('supporting_triples', []))}")

        # Session memory state
        session = result.get("session_context", {})
        print(f"\nSESSION MEMORY:")
        print(f"  Turn number: {session.get('turn_number', 0)}")
        print(f"  Entities accumulated: {session.get('entities_accumulated', 0)}")

        qa_results.append(result)

    all_results["qa_results"] = qa_results

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: KGAFE Evaluation of the QA answers
    # ──────────────────────────────────────────────────────────────────
    section("STEP 4: KGAFE EVALUATION — ATOMIC FACT VERIFICATION")

    from evaluation.kgafe.evaluator import KGAFEEvaluator

    evaluator = KGAFEEvaluator(
        kg=kg,
        model="gemma4:31b",
        enable_judge_panel=True,
    )

    kgafe_results = []
    for i, (question, qa_result) in enumerate(zip(DEMO_QUESTIONS, qa_results), 1):
        section(f"KGAFE EVALUATION — QUERY {i}")
        answer = qa_result.get("final_answer", "")
        print(f"Q: {question}")
        print(f"A: {answer[:300]}...\n")

        t0 = time.time()
        eval_result = evaluator.evaluate_answer(
            question=question,
            answer=answer,
        )
        elapsed = time.time() - t0

        m = eval_result.metrics
        print(f"\n{'─'*60}")
        print(f"KGAFE METRICS:")
        print(f"  KGAFE Score:        {m.kgafe_score:.4f}")
        print(f"  KG Faithfulness:    {m.kg_faithfulness:.4f}")
        print(f"  KG Precision:       {m.kg_precision:.4f}")
        print(f"  Hallucination Rate: {m.hallucination_rate:.4f}")
        print(f"  Path Validity:      {m.path_validity:.4f}")
        print(f"  Coverage:           {m.coverage:.4f}")
        print(f"  Groundedness:       {m.groundedness_score:.4f}")

        print(f"\n  Tier Distribution:")
        print(f"    Tier 1 (exact match):  {m.tier1_count}")
        print(f"    Tier 2 (path-based):   {m.tier2_count}")
        print(f"    Tier 3 (semantic):     {m.tier3_count}")

        print(f"\n  Verdict Distribution:")
        print(f"    Supported:          {m.supported_count}")
        print(f"    Contradicted:       {m.contradicted_count}")
        print(f"    Partially supported: {m.partially_supported_count}")
        print(f"    Unverifiable:       {m.unverifiable_count}")
        print(f"    Total facts:        {m.total_facts}")

        if eval_result.judge_verdict:
            jv = eval_result.judge_verdict
            print(f"\n  Judge Panel:")
            print(f"    Overall score: {jv.get('overall_score', 0):.2f}")
            print(f"    Pass: {jv.get('overall_pass', False)}")
            for iv in jv.get("individual_verdicts", []):
                print(f"    {iv.get('dimension', '?')}: {iv.get('score', 0):.2f} — {iv.get('reasoning', '')[:80]}")

        print(f"\n  Atomic Facts Breakdown:")
        for j, af in enumerate(eval_result.atomic_facts[:5], 1):
            vr = eval_result.verification_results[j-1] if j <= len(eval_result.verification_results) else {}
            verdict = vr.get("verdict", "?")
            tier = vr.get("tier", "?")
            conf = vr.get("confidence", 0)
            print(f"    {j}. \"{af.get('text', '?')[:80]}\"")
            print(f"       Verdict: {verdict} (Tier {tier}, conf={conf:.2f})")

        print(f"\n  Evaluation time: {elapsed:.1f}s")

        kgafe_results.append(eval_result.to_dict())

    all_results["kgafe_results"] = kgafe_results

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Aggregate KGAFE metrics
    # ──────────────────────────────────────────────────────────────────
    section("AGGREGATE KGAFE METRICS")

    n = len(kgafe_results)
    if n > 0:
        avg = lambda key: sum(r["metrics"].get(key, 0) for r in kgafe_results) / n
        print(f"  Questions evaluated:     {n}")
        print(f"  Avg KGAFE Score:         {avg('kgafe_score'):.4f}")
        print(f"  Avg KG Faithfulness:     {avg('kg_faithfulness'):.4f}")
        print(f"  Avg KG Precision:        {avg('kg_precision'):.4f}")
        print(f"  Avg Hallucination Rate:  {avg('hallucination_rate'):.4f}")
        print(f"  Avg Coverage:            {avg('coverage'):.4f}")
        print(f"  Avg Groundedness:        {avg('groundedness_score'):.4f}")

        total_facts = sum(r["metrics"]["verdict_distribution"]["total"] for r in kgafe_results)
        total_supported = sum(r["metrics"]["verdict_distribution"]["supported"] for r in kgafe_results)
        total_contradicted = sum(r["metrics"]["verdict_distribution"]["contradicted"] for r in kgafe_results)
        total_unverifiable = sum(r["metrics"]["verdict_distribution"]["unverifiable"] for r in kgafe_results)

        print(f"\n  Total atomic facts:      {total_facts}")
        print(f"  Total supported:         {total_supported} ({total_supported/max(total_facts,1):.1%})")
        print(f"  Total contradicted:      {total_contradicted} ({total_contradicted/max(total_facts,1):.1%})")
        print(f"  Total unverifiable:      {total_unverifiable} ({total_unverifiable/max(total_facts,1):.1%})")

        all_results["aggregate_kgafe"] = {
            "num_questions": n,
            "avg_kgafe_score": round(avg("kgafe_score"), 4),
            "avg_kg_faithfulness": round(avg("kg_faithfulness"), 4),
            "avg_kg_precision": round(avg("kg_precision"), 4),
            "avg_hallucination_rate": round(avg("hallucination_rate"), 4),
            "avg_coverage": round(avg("coverage"), 4),
            "avg_groundedness": round(avg("groundedness_score"), 4),
            "total_facts": total_facts,
            "total_supported": total_supported,
            "total_contradicted": total_contradicted,
            "total_unverifiable": total_unverifiable,
        }

    # ──────────────────────────────────────────────────────────────────
    # Save all results
    # ──────────────────────────────────────────────────────────────────
    section("SAVING RESULTS")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"All results saved to: {OUTPUT_PATH}")

    section("DEMO COMPLETE")
    print("What you just saw:")
    print("  1. Domain clustering: KG entities grouped into expert domains")
    print("  2. Advanced QA: Active exploration, debate, critic, provenance")
    print("  3. KGAFE: Each QA answer decomposed into atomic facts and verified")
    print(f"\nOpen {OUTPUT_PATH} for the full structured output.")


if __name__ == "__main__":
    main()
