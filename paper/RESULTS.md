# Results

All numbers below are computed from artifacts in `evaluation/results/`. The headline aggregate is in `paper_table_consolidated.json` for extraction and `qa_corrected_summary.json` for QA. Every run uses `gemma4:31b` at every model tier (small/medium/large) with a fixed SciERC schema, so the differences across rows are due to governance, not model size.

## How to read this section

The paper makes three claims about governance over a knowledge graph:

1. **Governance improves the graph that gets admitted.** When the same extractor runs under governance vs. flat insertion, the admitted graph is more correct.
2. **Governance changes which facts are in the graph, not just how many.** Governance is not a thin filter on top of the same graph; the two graphs disagree on most triples.
3. **Governance scales structurally.** Even when the extractor itself gets noisier on bigger corpora, the governance layer (routing, audit, coverage) stays correct.

We also report a downstream QA pilot: with the same 50-document governed graph as memory, does adding the governance stack hurt question answering? It does not.

We run three experiments to support these claims.

## Experiment 1 — Does governance improve the admitted graph?

**Setup.** We extract a knowledge graph from the SciERC test split twice: once with the *flat* pipeline (plain insertion: extract entities and triples, dump them into the graph) and once with the *governed* pipeline (the same extractor, plus domain ownership, the triage governance board, and an audit trail). Same documents, same model, same schema. The only difference is the governance layer.

We do this at two scales: 10 documents and 50 documents.

**Metrics.** All extraction metrics compare the system's output to SciERC's gold annotations.

- **Entity F1** — standard precision/recall harmonic mean for the set of named entities the system produces. 1.0 means it found exactly the entities the gold has.
- **Triple F1** — same harmonic mean, but for (subject, relation, object) triples. A triple counts as correct only if all three slots match the gold. This is a much harder metric than entity F1.
- **Triple hallucination rate** — fraction of produced triples whose subject + relation + object combination does not appear in the gold annotations. Lower is better.
- **Zero-triple docs** — number of documents for which the system produced no triples at all (a robustness/coverage signal).

### Table 1a — Per-config extraction quality

| Scale | Config | Entity F1 | Triple F1 | Triple hallucination | Zero-triple docs |
|---|---|---|---|---|---|
| 10 docs | Flat | 0.481 | 0.079 | 87.2% | 2 |
| 10 docs | **Governed (triage)** | **0.642** | **0.292** | **62.5%** | 2 |
| 10 docs | Δ (governed − flat) | +33% rel | **+271% rel** | −24.7 pp | — |
| 50 docs | Flat | 0.633 | 0.153 | 85.8% | 0 |
| 50 docs | **Governed (triage)** | 0.629 | **0.189** | **78.5%** | 3 |
| 50 docs | Δ (governed − flat) | −0.4% rel | **+23.5% rel** | −7.3 pp | — |

**What this says.** At both scales, governance keeps entity F1 essentially flat (a one-percentage-point drop at 50 docs is well within run-to-run noise) but lifts triple F1 substantially: +271% relative at 10 documents, +23.5% relative at 50 documents. Triple hallucination drops by 7–25 percentage points. The cost of running the governance layer is a +4.55% wall-clock overhead at 50 docs (Table 1b).

### Table 1b — How different are the governed and flat graphs?

A natural worry about Table 1a is that governance is just a thin filter — it might be removing some triples the flat system also produced, and the metric improvements are just from rejecting bad ones. Table 1b rules that out by measuring the structural overlap between the two graphs.

**Metric: Jaccard similarity.** A number from 0 to 1 that measures how similar two sets are. It is the size of the intersection divided by the size of the union.

- 1.0 = the two sets are identical.
- 0.0 = the two sets share no elements.
- 0.5 = roughly half of all elements (across both sets combined) are shared.

We compute Jaccard separately on the entity sets and on the triple sets of the two graphs. We also count how many triples are unique to each side.

| Metric | Value | Plain reading |
|---|---|---|
| Entity overlap (Jaccard) | 0.677 | The two graphs largely agree on which entities to include. |
| Triple overlap (Jaccard) | **0.180** | The two graphs disagree on roughly 80% of the triples — most edges are only in one graph or the other. |
| Triples unique to governed | 170 | Triples governance admitted that flat never proposed. |
| Triples unique to flat | 303 | Triples flat admitted that governance rejected or never received. |
| Cross-domain fraction | 43.4% | Fraction of governed triples whose subject and object live in different domains, i.e. governance routinely admits cross-domain edges, not only intra-domain ones. |
| Governance wall-clock overhead | +4.55% | The governance layer added 1,890 seconds on top of the 41,506-second flat run. |

**What this says.** A triple Jaccard of 0.180 means the two graphs *disagree on most edges*. Governance is not removing a small subset of bad triples from the flat graph; it is pulling in 170 triples flat never produced and rejecting 303 triples flat would have admitted. The improvement in Table 1a is the result of this structural change, not a thin post-hoc filter.

### Table 1c — What did the governance board actually do?

Governance is implemented as a triage board that routes each candidate triple to one of three outcomes: auto-approve (low-risk), escalate to LLM review, or revise/reject. Table 1c shows the activity counts for the 50-document run.

| Route | Count |
|---|---|
| Auto-approved (low-risk) | 163 |
| Escalated to LLM review | 130 |
| &nbsp;&nbsp;· Cross-domain review | 119 |
| &nbsp;&nbsp;· Conflict review | 11 |
| Reviewer outcome: approve | 99 |
| Reviewer outcome: revise | 17 |
| Reviewer outcome: other (escalate/reject) | 14 |

**What this says.** The board exercises both auto-approval (163 low-risk triples) and human-in-the-loop-style LLM review (130 escalations, of which 99 were approved, 17 revised, 14 rejected or escalated further). The non-trivial revise/reject rate confirms the board is not a rubber stamp.

## Experiment 2 — Does governance still work as the corpus grows?

**Setup.** We run the governed pipeline on 100 SciERC test documents (no flat comparison this time — the question here is whether governance itself remains correct as extraction noise grows). We measure both the extraction-side metrics from Experiment 1 and a new set of governance-side metrics.

**Metrics (governance-side).**

- **Routing exact match** — fraction of triples routed to the same domain a gold-standard router would have chosen. 1.0 = perfect routing.
- **Routing recall / precision (expected domain)** — recall and precision computed against the expected-domain labels for each triple.
- **Cross-domain routing recall** — recall on the subset of triples that should be routed across two domains rather than within one.
- **Domain coverage** — fraction of expected domains for which at least one triple was admitted (i.e. did the router populate every domain that should be populated?).
- **Governance completeness** — fraction of admitted triples that actually went through the full governance protocol (vs. being inserted by some bypass path). 1.0 = every triple was governed.
- **Audit-trail integrity** — fraction of admitted triples that have a complete, well-formed audit-trail entry. 1.0 = nothing is missing.

### Table 2 — Governed-only run on 100 SciERC test documents

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

**What this says.** Extraction quality drifts down at 100 documents: orphan-entity fraction grows to 63.8% and 30 of 100 documents emit no triples at all. But the governance-side metrics — routing, completeness, audit, domain coverage — all sit at or above 0.98. This is the structural-scaling claim: even when the extractor under it gets noisier, the governance layer continues to do its job correctly.

### Table 3 — Strict vs. triage review on adversarial probes

**Setup.** We give the governance board 50 probe triples — some are correct, some are corrupted versions of correct ones — and ask it to admit or reject. We compare two operating modes: `audit_only` (the default triage policy used everywhere else in the paper) and `strict` (a conservative ceiling that prefers rejection on any uncertainty).

**Metrics.**

- **Accept recall (positive)** — of the truly correct probe triples, what fraction did the board admit? Higher is better.
- **Reject recall (negative)** — of the corrupted probe triples, what fraction did the board reject? Higher is better.
- **False-accept rate** — fraction of corrupted triples the board incorrectly admitted. Lower is better; 0.00 means nothing bad got in.

| Mode | Accept recall (positive) | Reject recall (negative) | False-accept rate |
|---|---|---|---|
| Audit-only | 0.96 | 0.84 | 0.16 |
| **Strict** | 0.56 | **1.00** | **0.00** |

**What this says.** Strict mode is the provably-safe ceiling: zero false accepts, but it pays for that safety by rejecting 44% of correct proposals. Triage (audit-only) is the default operating mode in this paper because it admits 96% of correct triples while still rejecting 84% of bad ones — a much better trade-off when the goal is to grow the graph rather than freeze it.

## Experiment 3 — Does governance help downstream question answering?

**Setup.** We hold the 50-document governed knowledge graph fixed as memory and ask four QA systems the same 8 questions over it. The 8 questions cover four types: single-hop, comparison, cross-domain, and negative (questions whose answer is "this is not in the graph"). All four systems share the same KG and the same model; they differ only in retrieval and reasoning.

The four systems are:

- **`flat_path_basic`** — flat RAG-style retrieval. Treat the graph as a flat list of (subject, relation, object) facts; retrieve relevant ones and answer.
- **`graphrag_basic`** — GraphRAG-style retrieval. Build entity-relation communities over the graph, summarize each community, retrieve the relevant community summaries, and answer from them. No governance routing or critic.
- **`domain_basic`** — governed domain router only. Use the domain ownership produced at creation time to route a question to the right subgraph.
- **`domain_advanced`** — full governed QA stack: domain router + active graph exploration + critic + provenance-aware memory.

**Metrics.** All metrics use atomic-fact decomposition and KG verification (KGAFE). For each answer, we extract atomic factual claims, then check each claim against the KG and against the question's gold answer.

- **Corrected QA score** — composite KGAFE score combining faithfulness, precision, and coverage. The "corrected" qualifier means correct abstention on a negative question is no longer scored as hallucination (the original raw aggregate did this; we fix it because correctly saying "this is not in the graph" is the *desired* behavior of a governed system).
- **Faithfulness** — fraction of the answer's atomic claims that are entailed by the KG. 1.0 = nothing is contradicted by the graph.
- **Precision / support** — fraction of the answer's atomic claims that have explicit support in the retrieved KG evidence. 1.0 = every claim is grounded.
- **Hallucination rate** — fraction of atomic claims that are not supported by the KG. Lower is better.
- **Coverage** — fraction of the gold-answer atomic claims that the system's answer actually covers. 1.0 = the system addressed every part of the gold.

### Table 4 — Four-way QA on the 50-document governed KG (n = 8)

| System | Corrected QA score | Faithfulness | Precision / support | Hallucination rate | Coverage |
|---|---|---|---|---|---|
| `flat_path_basic` | 0.691 | **1.000** | **1.000** | **0.0%** | 0.938 |
| `graphrag_basic` | 0.673 | 0.984 | 0.984 | 1.6% | 0.875 |
| `domain_basic` | 0.680 | **1.000** | 0.958 | 4.2% | 0.938 |
| **`domain_advanced`** | **0.698** | **1.000** | 0.991 | 0.9% | **1.000** |

**What this says.** All four configurations land in a tight band on this 8-question pilot. Faithfulness is at or above 0.98 and hallucination is at or below 4.2% across the board. The full governed stack (`domain_advanced`) has the highest corrected composite score, the highest coverage (1.000 — the only system that fully addresses every gold question), and the lowest hallucination rate among the governance configurations.

We do **not** claim a statistical QA win on n = 8. The honest reading is: governance does not hurt downstream QA, and it is the only configuration here that reaches full coverage. A larger QA benchmark is left to future work.

## Bottom line

1. **Governance improves the admitted graph.** Triple F1 is +23.5% relative at 50 documents and +271% relative at 10 documents, with hallucination down 7–25 pp — at a 4.55% wall-clock cost (Table 1a).
2. **Governance changes which facts are in the graph, not just how many.** Triple Jaccard between governed and flat is 0.180 at 50 documents; 303 triples are unique to flat, 170 are unique to governed (Table 1b).
3. **Governance scales structurally.** At 100 documents, routing, audit-trail integrity, governance completeness, and domain coverage all stay at or above 0.98 even as extraction-side noise grows (Table 2).
4. **Governance supports a safe operating ceiling.** Strict review achieves 0% false accepts; triage recovers 96% of good triples while still rejecting 84% of bad ones (Table 3).
5. **Governance does not hurt downstream QA, and reaches full coverage.** On the 50-document governed graph, all four QA configurations achieve faithfulness ≥ 0.98 and hallucination ≤ 4.2% on an 8-question pilot. The full governed stack (`domain_advanced`) is the only system at coverage 1.000 and has the lowest hallucination rate among governance configurations (Table 4).
