# Results Section Working State
Updated: 2026-05-04T00:00:00
## Extraction / Governance Construction
| Condition | Triples | Strict Triple F1 | Fuzzy/Mapped Triple F1 | Hallucination | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| GPT-5 Flat KG | 1366 | 0.116 | 0.204 | 92.7% | 100 / 1266 / 251 |
| Domain Rules Only | 978 | 0.131 | 0.262 | 91.1% | 87 / 891 / 264 |
| GPT-5 MaKG Neighborhood | 750 | 0.144 | 0.305 | 89.5% | 79 / 671 / 272 |
| Global Reviewer Minimal | 674 | 0.146 | 0.304 | 88.9% | 75 / 599 / 276 |
| Global Reviewer Matched | 533 | 0.167 | 0.330 | 86.1% | 74 / 459 / 277 |
| GPT-5 MaKG Domain-Memory Reviewer | 472 | 0.173 | 0.333 | 85.0% | 71 / 401 / 280 |

Interpretation: MaKG beats flat and deterministic domain-rules-only on F1 and hallucination. A simple centralized approve/reject reviewer is roughly tied with basic MaKG (0.304 vs 0.305 fuzzy/mapped F1), but the domain-memory MaKG reviewer is strongest overall: 0.173 strict F1, 0.333 fuzzy/mapped F1, and 85.0% hallucination. This supports the paper claim that governance matters most when domain experts carry domain-local memory rather than acting as generic LLM filters.

## Evidence-Only Ablation
| Condition | Docs | Triples | Strict Triple F1 | Fuzzy/Mapped Triple F1 | Hallucination | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Flat insertion | 10 | 305 | 0.163 | 0.265 | 89.5% | 32 / 273 / 56 |
| EvidenceLinker + Verification only | 10 | 298 | 0.124 | 0.228 | 91.9% | 24 / 274 / 64 |
| MaKG Neighborhood | 10 | 166 | 0.165 | 0.362 | 87.3% | 21 / 145 / 67 |

Interpretation: evidence filtering alone does not explain the MaKG gain. MaKG admits fewer triples, has lower hallucination, and has the best fuzzy/mapped F1 on the 10-doc Study 1 slice.

## QA
| System / Setup | Dataset | F1 | Read |
|---|---|---:|---|
| GPT-5 RAG | MuSiQue n=50 supporting-only | 0.900 | strong reading-comprehension baseline |
| GPT-5 GraphRAG | MuSiQue n=50 supporting-only | 0.709 | graph-aware baseline |
| GPT-5 Flat KG QA | MuSiQue n=50 supporting-only | 0.718 | graph without governed routing |
| GPT-5 MaKG domain QA | MuSiQue n=50 supporting-only | 0.775 | best graph-routed system, but not above oracle-like RAG |
| Gemma RAG | MuSiQue n=50 full distractors | 0.374 | RAG drops from 0.803 supporting-only to 0.374 with distractors |

## Pending
- Human-review CSV exists and should be annotated externally.
- Cost/runtime table should be added from usage logs.
- Full-distractor governed QA is not run; use RAG distractor degradation as motivation, not a governed win.
