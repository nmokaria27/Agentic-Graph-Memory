# Validation Study 1 Preliminary Result
Updated: 2026-05-04T00:00:00

Question: is MaKG only better because EvidenceLinker/Verification filters unsupported triples?

Setup: GPT-5, SciERC test first 10 docs, fixed SciERC schema. Study 1 condition enables EvidenceLinker + strict source-only VerificationAgent, then inserts approved triples into a flat KG with no domain ownership or governance.

| Condition | Docs | Predicted triples on slice | Evidence-linked triples | Strict Triple F1 | Fuzzy/Mapped Triple F1 | Hallucination | TP / FP / FN | Runtime |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Flat insertion (no evidence/governance) | 10 | 305 | not measured | 0.163 | 0.265 | 89.5% | 32 / 273 / 56 |  |
| Study 1: Evidence+verification only, flat insert | 10 | 298 | 298/298 | 0.124 | 0.228 | 91.9% | 24 / 274 / 64 | 821s |
| MaKG neighborhood governance | 10 | 166 | via audit log | 0.165 | 0.362 | 87.3% | 21 / 145 / 67 |  |

Read: evidence filtering alone is not the whole MaKG story. Evidence+verification-only flat insertion admits nearly as many triples as flat insertion, but it has lower strict F1, lower fuzzy/mapped F1, and higher hallucination than MaKG. MaKG admits fewer triples, lowers hallucination, and has the best strict and fuzzy/mapped F1 on this slice. This supports the claim that governance is doing more than source filtering: ownership/routing/admission changes what enters memory.

Caveat: n=10 is still a preliminary ablation slice. It is enough to show the design and direction of Study 1; a 50-doc evidence-only run is optional and should only be run if runtime/cost allows.
