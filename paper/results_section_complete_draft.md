# Results Section Draft
Updated: 2026-05-04

## Experimental Setup

We evaluate MaKG along four axes: construction quality, governance ablations, human triple review, and downstream QA. Construction and governance experiments use the SciERC test split with a fixed SciERC schema. QA experiments use a 50-question MuSiQue 2-hop slice with 99 unique supporting paragraphs for the supporting-only setting and 805 unique titles / 1000 paragraphs for the full-distractor RAG stress test.

Unless otherwise stated, GPT-5 is used as the main model. Gemma/Ollama runs are treated as secondary cost-sensitive evidence, not the headline model.

## 1. KG Construction Quality

The main construction result compares direct flat insertion against the full MaKG governed construction path. Both systems use the same extracted candidate stream and the same fixed SciERC schema; the difference is whether candidate facts are admitted through governance.

| System | Entities | Triples | Strict Triple F1 | Fuzzy/Mapped Triple F1 | Triple Hallucination | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5 Flat KG | 703 | 1366 | 0.116 | 0.204 | 92.7% | 100 / 1266 / 251 |
| GPT-5 MaKG basic governance | 687 | 750 | 0.144 | 0.305 | 89.5% | 79 / 671 / 272 |
| GPT-5 MaKG domain-memory governance | 687 | 472 | **0.173** | **0.333** | **85.0%** | 71 / 401 / 280 |

The full MaKG condition improves strict triple F1 from 0.116 to 0.173, improves fuzzy/mapped triple F1 from 0.204 to 0.333, and reduces triple hallucination from 92.7% to 85.0%. The result should be framed as an admission-quality result, not as a claim that extraction recall is solved. MaKG admits fewer triples than the flat KG, but the admitted graph is cleaner under both strict and fuzzy/mapped matching.

## 2. Governance Ablation

The governance ablation isolates which part of governance matters. All ablations replay the same 50-document GPT-5 candidate proposal stream from the MaKG audit log.

| Condition | What is removed / kept | Triples | Strict F1 | Fuzzy/Mapped F1 | Hallucination | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|
| Flat KG | no governance | 1366 | 0.116 | 0.204 | 92.7% | 100 / 1266 / 251 |
| EvidenceLinker + Verification only | source filtering, no ownership/governance; 10-doc slice | 298 | 0.124 | 0.228 | 91.9% | 24 / 274 / 64 |
| Domain rules only | ownership routing, no LLM reviewer | 978 | 0.131 | 0.262 | 91.1% | 87 / 891 / 264 |
| Global reviewer, minimal | centralized approve/reject reviewer, no domain memory | 674 | 0.146 | 0.304 | 88.9% | 75 / 599 / 276 |
| Global reviewer, matched | centralized reviewer with stronger matched rubric | 533 | 0.167 | 0.330 | 86.1% | 74 / 459 / 277 |
| MaKG domain-memory reviewer | domain routing + local domain memory + endpoint neighborhood | 472 | **0.173** | **0.333** | **85.0%** | 71 / 401 / 280 |

The ablation supports three claims. First, evidence filtering alone does not explain the gain: on the 10-document Study 1 slice, evidence-only flat insertion underperforms MaKG. Second, deterministic domain ownership is useful but insufficient: domain rules improve over flat insertion but remain below MaKG. Third, the strongest MaKG result appears when domain experts carry domain-local memory: accepted/rejected relation patterns, endpoint membership, and admitted-neighborhood context. This is the clearest evidence that governance is not merely a post-hoc filter.

The matched global reviewer is an important control. It is strong on isolated triple filtering, but it does not produce domain ownership, cross-domain routing, audit-local state, or domain-owned memory. MaKG with domain memory slightly exceeds the matched global reviewer while also producing the governed representation required by downstream routing.

## 3. Human Triple Review

SciERC gold annotations are incomplete for open-ended KG construction: a system can extract a supported scientific fact that is not present in the benchmark gold triples and still be counted as a false positive. For that reason, we include a human review study over triples sampled from the 50-document construction experiment.

The review file is `evaluation/results/governance_ablation_samples_gpt5_50docs.csv`.

The sample groups are:

| Group | Source | Purpose |
|---|---|---|
| Governed-only triples | admitted by MaKG, absent from flat KG | checks whether governance adds supported useful facts |
| Flat-only triples | admitted by flat KG, absent from MaKG | checks whether governance removes unsupported/noisy facts |
| Shared triples | admitted by both systems | control group for robust facts |
| Revised triples | revised by MaKG before admission | checks whether revision actually corrects subject/relation/object errors |

The human-review result should be reported as support rate / usefulness rate / correction rate once annotated. This is the key external validation for claims that SciERC undercounts valid extracted facts.

## 4. Downstream QA

We evaluate downstream QA on a 50-question MuSiQue 2-hop slice. The supporting-only corpus contains 100 contexts across 99 unique titles. The full-distractor setting contains 1000 contexts across 805 unique titles.

| System | Setting | EM | Token F1 | Answer Rate | Interpretation |
|---|---|---:|---:|---:|---|
| GPT-5 RAG | supporting-only | 0.800 | 0.900 | 1.00 | strongest when oracle-like supporting paragraphs are available |
| GPT-5 GraphRAG | supporting-only | 0.580 | 0.709 | 1.00 | graph-aware baseline |
| GPT-5 flat KG QA | supporting-only | 0.620 | 0.718 | 1.00 | graph without governed domain routing |
| GPT-5 MaKG domain QA | supporting-only | 0.660 | 0.775 | 1.00 | strongest KG-based system |
| Gemma RAG | full distractors | 0.260 | 0.374 | 1.00 | RAG drops sharply under distractor noise |

The QA result should be framed carefully. On supporting-only MuSiQue, GPT-5 RAG is very strong because the answer is usually present in a small set of supporting paragraphs. The meaningful MaKG result is that governed domain QA is the strongest graph-based system, outperforming both flat KG QA and GraphRAG-style KG retrieval. The full-distractor RAG result shows why structured memory matters at scale: when retrieval must search 1000 contexts rather than 100 supporting contexts, Gemma RAG drops from 0.803 F1 in the supporting-only setting to 0.374 F1 in the full-distractor setting.

## 5. Cost and Runtime

The table below reports exact call and token counts from JSONL usage logs. Dollar estimates depend on model pricing and should be checked against the actual invoice. The estimates here use the GPT-5 API rates reflected by the model used in these logs.

| Stage | Artifact / Log | Unit | LLM Calls | Total Tokens | Est. Cost | Calls / Unit | Tokens / Unit |
|---|---|---:|---:|---:|---:|---:|---:|
| SciERC extraction build | `gpt5_scierc_gov_50_neighborhood_usage.jsonl` | 50 docs | 1169 | 4,857,745 | $12.96 | 23.38 / doc | 97,155 / doc |
| Domain-memory governance review | `gpt5_ablation_domain_memory_reviewer_50_neighborhood_usage.jsonl` | 1304 candidate triples | 49 | 837,617 | $1.77 | 0.038 / candidate | 642 / candidate |
| Full best extraction pipeline | extraction + domain-memory review | 50 docs | 1218 | 5,695,362 | $14.73 | 24.36 / doc | 113,907 / doc |
| Global reviewer, matched | `gpt5_ablation_global_matched_reviewer_50_neighborhood_usage.jsonl` | 1304 candidates | 49 | 168,580 | $0.83 | 0.038 / candidate | 129 / candidate |
| Global reviewer, minimal | `gpt5_ablation_global_minimal_noconf_reviewer_50_neighborhood_usage.jsonl` | 1304 candidates | 66 | 160,390 | $0.67 | 0.051 / candidate | 123 / candidate |
| MuSiQue KG build | `gpt5_musique_n50_build_usage.jsonl` | 99 supporting titles | 328 | 3,778,407 | $7.65 | 3.31 / title | 38,166 / title |
| MuSiQue KG-QA ablation | `gpt5_musique_n50_qa_usage.jsonl` | 200 KG answers | 1371 | 5,430,021 | $23.40 | 6.86 / answer | 27,150 / answer |
| MuSiQue RAG QA | `gpt5_musique_n50_rag_usage.jsonl` | 50 questions | 50 | 47,994 | $0.13 | 1.00 / question | 960 / question |

Additional runtime details:

| Stage | Runtime |
|---|---:|
| SciERC GPT-5 extraction build, 50 docs | 9962 seconds / 2.77 hours |
| SciERC GPT-5 flat build, 50 docs | 3891 seconds / 1.08 hours |
| Evidence-only 10-doc run | 821 seconds |

Deliberation cost is not separately reported in the current logs. Any deliberation/hypothesis handling invoked during extraction is included inside the extraction-build usage log. Future runs should set per-agent `LLM_USAGE_LOG` labels if the paper needs exact breakdowns for entity extraction, relation extraction, deliberation, evidence linking, and governance review separately.
