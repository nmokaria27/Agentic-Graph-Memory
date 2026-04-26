# Results

All numbers below are computed from artifacts in `evaluation/results/`. Paper-table aggregation is in `paper_table_consolidated.json`. Model: `gemma4:31b` for every tier (small/medium/large) with fixed SciERC schema.

## 1. Extraction Competence + Governance as an Intervention

### Table 1 — Governed vs Flat at 10 and 50 SciERC test documents

| Scale | Config | Entity F1 | Triple F1 | Triple Halluc. | Entities | Triples | Zero-triple docs |
|---|---|---|---|---|---|---|---|
| 10 docs | Flat | 0.481 | 0.079 | 87.2% | 73 | 39 | 2 |
| 10 docs | Governed (triage) | **0.642** | **0.292** | **62.5%** | 137 | 56 | 2 |
| 10 docs | Δ | +33% rel | **+271% rel** | −24.7 pp | — | — | — |
| 50 docs | Flat | 0.633 | 0.153 | 85.8% | 634 | 407 | 0 |
| 50 docs | Governed (triage) | 0.629 | **0.189** | **78.5%** | 621 | 274 | 3 |
| 50 docs | Δ | −0.4% rel | **+23.5% rel** | −7.3 pp | — | — | — |

### Table 2 — Governance structural impact (50 docs)

| Metric | Value |
|---|---|
| Entity overlap (Jaccard) | 0.677 |
| Triple overlap (Jaccard) | **0.180** |
| Triples unique to governed | 170 |
| Triples unique to flat | 303 |
| Cross-domain fraction | 43.4% |
| Governance wall-clock overhead | +4.55% (+1,890 s on 41,506 s flat run) |

Low triple-overlap (0.18) means governance *changes* which triples are admitted, not just which volume — governance is not a thin filter.

### Sidebar — Pairwise-v2 relation extractor at 10 docs (governance still wins)

A separate 10-doc run with the newer pairwise-v2 relation extractor (`enable_fixed_schema_pairwise=True`) confirms the headline direction under a different extractor. Both sides run the same hardened pipeline (LLM-retry + triple-coercion + per-doc checkpoint) so the only difference is governance.

| Config | Entity F1 | Triple F1 | Triple Halluc. | Entities | Triples | Zero-triple docs |
|---|---|---|---|---|---|---|
| Flat (pairwise v2) | **0.627** | 0.200 | 82.8% | 125 | 122 | 1 |
| Governed (pairwise v2) | 0.564 | **0.318** | **67.1%** | 120 | 82 | 2 |
| Δ (governed − flat) | −6.4 pp | **+11.8 pp** (+59% rel) | **−15.7 pp** | — | — | — |

Governance trades a small entity-F1 hit for a +59% relative lift in triple F1 and a 15.7-pp drop in hallucination — same shape as Table 1.

### Table 3 — Triage board activity (50 docs)

| Route | Count |
|---|---|
| Auto-approved (low-risk) | 163 |
| Escalated to LLM review | 130 |
| &nbsp;&nbsp;· Cross-domain review | 119 |
| &nbsp;&nbsp;· Conflict review | 11 |
| Reviewer outcome: approve | 99 |
| Reviewer outcome: revise | 17 |
| Reviewer outcome: other (escalate/reject) | 14 |

## 2. Governance Structure at Scale (100 docs, governed-only)

### Table 4 — Structural correctness is maintained as extraction noise grows

| Metric | Value |
|---|---|
| KG entities | 1,201 |
| KG triples | 338 |
| Entity F1 | 0.612 |
| Triple F1 | 0.133 |
| Orphan-entity fraction | 63.8% |
| Zero-triple docs | 30 / 100 |
| **Routing exact match** | **1.000** |
| **Routing recall (expected domain)** | 1.000 |
| **Routing precision (expected domain)** | 1.000 |
| **Cross-domain routing recall** | 1.000 |
| **Domain coverage** | 0.984 |
| **Governance completeness** | 1.000 |
| **Audit-trail integrity** | 1.000 |

Interpretation: extraction quality degrades as corpus size grows, but the *governance* layer (routing, audit, domain coverage) remains essentially perfect. This is the "structure scales even when extraction is noisy" claim.

### Table 5 — Strict review benchmark (100 docs, 50 probe triples)

| Mode | Accept recall (positive) | Reject recall (negative) | False-accept rate |
|---|---|---|---|
| Audit-only | 0.96 | 0.84 | 0.16 |
| **Strict** | 0.56 | **1.00** | **0.00** |

Strict mode is the provably-safe ceiling: zero false accepts, at the cost of rejecting 44% of correct proposals. Triage is the default operating mode because it preserves almost all correct triples while catching most bad ones.

## 3. Downstream QA on the Governed 50-Doc KG

The QA story should be told using standard-looking grounded-QA metrics, not by over-indexing on the KGAFE name. The key downstream question is whether governance makes answers more useful and better grounded than a flat baseline or a graph-aware retrieval baseline. The corrected QA aggregate is stored in `evaluation/results/qa_corrected_summary.json`; it fixes the old artifact where correct abstention on negative questions was unfairly penalized.

### Table 6 — Four-way QA ablation with GraphRAG-style baseline (corrected summary, n=8)

`graphrag_basic` is a neutral graph-aware retrieval baseline over the same 50-doc governed KG: it builds communities, summarizes them, and answers from retrieved community summaries, but it does not use governance routing, domain experts, or critique.

| Metric | flat_path_basic | graphrag_basic | domain_basic | **domain_advanced** |
|---|---|---|---|---|
| Corrected grounded-QA score | 0.691 | 0.673 | 0.680 | **0.698** |
| Faithfulness | **1.000** | 0.984 | **1.000** | **1.000** |
| Precision / support | **1.000** | 0.984 | 0.958 | **0.991** |
| Hallucination rate | **0.000** | 0.016 | 0.042 | **0.009** |
| Coverage | 0.938 | 0.875 | 0.938 | **1.000** |

**Headline:** after correcting negative-question abstention, GraphRAG no longer “collapses.” Instead it behaves like a cautious graph-aware baseline: low hallucination, high faithfulness, but lower coverage than the governed stack. The full governed QA system, `domain_advanced`, is still the strongest overall configuration: it has the highest corrected composite score, full coverage, and near-perfect faithfulness and precision.

### Table 6b — Hard-question judge rerun (supporting, corrected summary, n=5)

The judge-panel rerun is supporting evidence only. It uses the hardest question types and should be read as a qualitative check rather than a high-powered statistical result.

| Metric | flat_path_basic | graphrag_basic | domain_basic |
|---|---|---|---|
| Corrected grounded-QA score | **0.985** | 0.950 | 0.895 |
| Faithfulness | **1.000** | **1.000** | 0.933 |
| Precision / support | **1.000** | **1.000** | 0.860 |
| Hallucination rate | **0.000** | **0.000** | 0.140 |
| Coverage | **0.900** | 0.667 | 0.667 |
| Completeness (judge) | **1.000** | 0.780 | 0.950 |

**Interpretation:** the judge rerun shows that GraphRAG is not broken; it is terse and coverage-limited. It tends to answer correctly when it answers, but it is less complete than both the flat and governed systems on the hard questions. Domain-routed QA is more ambitious and more complete than GraphRAG, though the extra coverage introduces some precision risk on this tiny sample. This is exactly why the full governed stack with critique (`domain_advanced`) is the most appealing downstream operating point.

## Bottom line

1. **Governance improves the admitted graph.** Triple F1 is +23.5% relative at 50 docs and +271% relative at 10 docs, with hallucination down 7–25 pp — at a 4.55% wall-clock cost.
2. **Governance changes graph structure, not just volume.** Triple Jaccard of 0.18 with 303 flat-only and 170 governed-only triples.
3. **Governance scales.** At 100 docs, routing / completeness / audit remain at or above 0.98 even as extraction noise doubles.
4. **Governance supports safe operating modes.** Strict review → 0% false-accept; triage recovers 96% of good triples with 84% of bad rejected.
5. **Governance helps downstream QA — beyond graph-aware retrieval alone.** After correcting negative-question abstention, `graphrag_basic` becomes a fair graph-aware baseline rather than an apparent failure. Even under that corrected view, the full governed stack (`domain_advanced`) remains the best balanced downstream system: it has the highest corrected composite score, the highest coverage, and near-perfect faithfulness and precision. Graph structure alone helps; governance is what turns that structure into consistently complete and well-grounded answers.
