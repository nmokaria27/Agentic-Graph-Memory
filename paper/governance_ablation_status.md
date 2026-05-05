# Governance Ablation Study Status
Updated: 2026-05-04T00:00:00

## Study 1: EvidenceLinker + Verification only
- Status: complete on 10 docs.
- Artifact: `paper/validation_study1_preliminary_example.md`.
- Read: evidence filtering alone does not explain MaKG. On the 10-doc slice, MaKG has higher strict F1, higher fuzzy/mapped F1, and lower hallucination than evidence+verification-only flat insertion.

| Condition | Docs | Triples | Strict F1 | Fuzzy/Mapped F1 | Hallucination | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Flat insertion | 10 | 305 | 0.163 | 0.265 | 89.5% | 32 / 273 / 56 |
| EvidenceLinker + Verification only | 10 | 298 | 0.124 | 0.228 | 91.9% | 24 / 274 / 64 |
| MaKG neighborhood governance | 10 | 166 | 0.165 | 0.362 | 87.3% | 21 / 145 / 67 |

## Study 1 Main Construction/Ablation Table
| Condition | Status | Triples | Strict F1 | Fuzzy/Mapped F1 | Hallucination | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|
| Flat insertion | complete | 1366 | 0.116 | 0.204 | 92.7% | 100 / 1266 / 251 |
| Domain rules only (new neighborhood audit) | complete | 978 | 0.131 | 0.262 | 91.1% | 87 / 891 / 264 |
| MaKG neighborhood governance | complete | 750 | 0.144 | 0.305 | 89.5% | 79 / 671 / 272 |
| Global reviewer, minimal approve/reject | complete | 674 | 0.146 | 0.304 | 88.9% | 75 / 599 / 276 |
| Global reviewer matched (new neighborhood audit) | complete | 533 | 0.167 | 0.330 | 86.1% | 74 / 459 / 277 |
| MaKG domain-memory reviewer | complete | 472 | 0.173 | 0.333 | 85.0% | 71 / 401 / 280 |

## Interpretation
- MaKG neighborhood beats flat insertion: strict F1 0.144 vs 0.116, fuzzy/mapped F1 0.305 vs 0.204, hallucination 89.5% vs 92.7%.
- MaKG neighborhood beats deterministic domain-rules-only: strict F1 0.144 vs 0.131, fuzzy/mapped F1 0.305 vs 0.262, hallucination 89.5% vs 91.1%.
- Minimal global reviewer is essentially tied with MaKG on pure SciERC extraction: MaKG fuzzy/mapped F1 0.305 vs minimal global reviewer 0.304. The difference is that MaKG produces governed memory: domain ownership, routing, audit records, cross-domain state, and downstream domain expert routing.
- Global reviewer matched is the strongest reviewer-only quality control: strict F1 0.167, fuzzy/mapped F1 0.330, hallucination 86.1%. This should be framed honestly as a strong-control or appendix result: a single strong reviewer can be an effective triple filter, but it still lacks ownership, routing, domain-local audit, cross-domain review state, and governed memory.
- MaKG domain-memory reviewer is now the strongest governance row: strict F1 0.173, fuzzy/mapped F1 0.333, hallucination 85.0%. This is the cleanest evidence that domain experts become useful when they actually carry domain-local memory: relation signatures, endpoint membership, prior accepted/rejected patterns, and admitted-neighborhood context.
- EvidenceLinker+Verification-only does not explain the MaKG lift. On 10 docs, it is worse than MaKG on strict F1, fuzzy/mapped F1, and hallucination.

## Pending Choices
- Human triple review CSV is ready and remains the best external validation against SciERC incompleteness.
- Cost/runtime table should be added as Study 3. Use GPT-5 call/token logs from the MaKG, evidence-only, and global-reviewer runs.
- Optional only if time remains: rerun evidence-only at 50 docs. The 10-doc result is already enough for the mentor's requested preliminary Study 1 table.
