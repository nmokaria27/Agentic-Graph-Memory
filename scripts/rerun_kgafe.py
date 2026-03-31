#!/usr/bin/env python3
"""
Re-run KGAFE evaluation on the existing QA answers from demo_results.json
using the fixed entity resolution code.

Compares old vs new results side-by-side.
"""

import json
import time

from multi_agent_kg.core.kg_operations import load_kg
from evaluation.kgafe.evaluator import KGAFEEvaluator


def main():
    # Load the KG
    print("Loading KG...")
    kg = load_kg("kg_export.json")
    print(f"  {len(kg.entities)} entities, {len(kg.triples)} triples\n")

    # Load existing demo results
    with open("demo_results.json") as f:
        demo = json.load(f)

    questions_and_answers = []
    for qa in demo["qa_results"]:
        questions_and_answers.append({
            "question": qa["question"],
            "answer": qa["final_answer"],
        })

    old_kgafe = demo["kgafe_results"]

    # Initialize KGAFE evaluator
    print("Initializing KGAFE evaluator...")
    evaluator = KGAFEEvaluator(
        kg=kg,
        model="gemma3:27b",
        enable_judge_panel=True,
    )

    # Run evaluation
    new_results = []
    for i, qa in enumerate(questions_and_answers):
        print(f"\n{'='*70}")
        print(f"QUESTION {i+1}: {qa['question']}")
        print(f"{'='*70}")
        print(f"Answer: {qa['answer'][:200]}...\n")

        t0 = time.time()
        eval_result = evaluator.evaluate_answer(
            question=qa["question"],
            answer=qa["answer"],
        )
        elapsed = time.time() - t0

        m = eval_result.metrics
        md = m.to_dict()
        old_m = old_kgafe[i]["metrics"] if i < len(old_kgafe) else {}

        print(f"\n{'─'*60}")
        print(f"KGAFE METRICS (new vs old):")
        print(f"  KGAFE Score:        {m.kgafe_score:.4f}  (was {old_m.get('kgafe_score', '?')})")
        print(f"  KG Faithfulness:    {m.kg_faithfulness:.4f}  (was {old_m.get('kg_faithfulness', '?')})")
        print(f"  KG Precision:       {m.kg_precision:.4f}  (was {old_m.get('kg_precision', '?')})")
        print(f"  Hallucination Rate: {m.hallucination_rate:.4f}  (was {old_m.get('hallucination_rate', '?')})")
        print(f"  Coverage:           {m.coverage:.4f}  (was {old_m.get('coverage', '?')})")
        print(f"  Groundedness:       {m.groundedness_score:.4f}  (was {old_m.get('groundedness_score', '?')})")
        print(f"  Correctness:        {m.correctness_score:.4f}  (was {old_m.get('correctness_score', '?')})")
        print(f"  Completeness:       {m.completeness_score:.4f}  (was {old_m.get('completeness_score', '?')})")
        print(f"  Judge Overall:      {m.judge_overall:.4f}  (was {old_m.get('judge_overall', '?')})")

        old_td = old_m.get("tier_distribution", {})
        print(f"\n  Tier Distribution:")
        print(f"    Exact match:  {m.tier1_count:3d}  (was {old_td.get('exact_match', '?')})")
        print(f"    Path-based:   {m.tier2_count:3d}  (was {old_td.get('path_based', '?')})")
        print(f"    Semantic:     {m.tier3_count:3d}  (was {old_td.get('semantic', '?')})")

        old_vd = old_m.get("verdict_distribution", {})
        print(f"\n  Verdict Distribution:")
        print(f"    Supported:    {m.supported_count:3d}  (was {old_vd.get('supported', '?')})")
        print(f"    Contradicted: {m.contradicted_count:3d}  (was {old_vd.get('contradicted', '?')})")
        print(f"    Unverifiable: {m.unverifiable_count:3d}  (was {old_vd.get('unverifiable', '?')})")
        print(f"    Total:        {m.total_facts:3d}  (was {old_vd.get('total', '?')})")

        print(f"\n  Evaluation time: {elapsed:.1f}s")

        new_results.append(eval_result.to_dict())

    # Aggregate
    n = len(new_results)
    if n > 0:
        print(f"\n{'='*70}")
        print(f"AGGREGATE COMPARISON ({n} questions)")
        print(f"{'='*70}")

        avg = lambda key: sum(r["metrics"].get(key, 0) for r in new_results) / n
        old_agg = demo.get("aggregate_kgafe", {})

        metrics = [
            ("avg_kgafe_score", "kgafe_score"),
            ("avg_kg_faithfulness", "kg_faithfulness"),
            ("avg_hallucination_rate", "hallucination_rate"),
            ("avg_coverage", "coverage"),
            ("avg_groundedness", "groundedness_score"),
        ]

        for agg_key, detail_key in metrics:
            new_val = avg(detail_key)
            old_val = old_agg.get(agg_key, "?")
            delta = ""
            if isinstance(old_val, (int, float)):
                d = new_val - old_val
                delta = f"  ({'+' if d >= 0 else ''}{d:.4f})"
            print(f"  {agg_key:30s}: {new_val:.4f}  (was {old_val}){delta}")

        total_facts = sum(r["metrics"]["verdict_distribution"]["total"] for r in new_results)
        total_supported = sum(r["metrics"]["verdict_distribution"]["supported"] for r in new_results)
        total_contradicted = sum(r["metrics"]["verdict_distribution"]["contradicted"] for r in new_results)
        total_unverifiable = sum(r["metrics"]["verdict_distribution"]["unverifiable"] for r in new_results)

        print(f"\n  Total atomic facts:      {total_facts}  (was {old_agg.get('total_facts', '?')})")
        print(f"  Total supported:         {total_supported}  (was {old_agg.get('total_supported', '?')})")
        print(f"  Total contradicted:      {total_contradicted}  (was {old_agg.get('total_contradicted', '?')})")
        print(f"  Total unverifiable:      {total_unverifiable}  (was {old_agg.get('total_unverifiable', '?')})")

    # Save new results
    output = {
        "kgafe_results": new_results,
        "aggregate_kgafe": {
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
        },
    }
    with open("kgafe_results_fixed.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to kgafe_results_fixed.json")


if __name__ == "__main__":
    main()
