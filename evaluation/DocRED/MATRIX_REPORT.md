# DocRED Experiment Matrix — Meta Report (2026-07-03, post coref-fix)

3 strategies × 2 slices, Re-DocRED dev. **Slice A** = docs 0–4 (diagnostic; every fix
was developed against these). **Slice B** = docs 30–34 (held-out; never hand-read,
never tuned on — the memorization check). Zero crashes, zero empty LLM calls across
all 25 doc-runs. Raw per-doc records: `evaluation/results/docred_kg_cache*/`;
scores: `evaluation/results/docred_scores_<strategy>_<slice>.json`.

## Headline numbers

| strategy | slice | entP | entR | pairR | flips | relF1@0.6 |
|---|---|---|---|---|---|---|
| rhf (pre-fix, last night) | A | 0.88 | 0.47 | 0.14 | 2 | 0.127 |
| **rhf (coref fixed)** | A | 0.89 | **0.56** | **0.17** | 1 | **0.191** |
| rhf (coref fixed) | B | 0.82 | 0.60 | 0.11 | 1 | 0.043 |
| singlepass | A | 0.87 | **0.81** | **0.23** | 1 | 0.188 |
| singlepass | B | 0.79 | **0.88** | **0.20** | 3 | 0.133 |
| hybrid v1 | A | 0.83 | 0.61 | 0.19 | 7 | 0.156 |
| hybrid v1 | B | 0.72 | 0.66 | 0.12 | 3 | 0.100 |

## Verdict 1 — the coref fix is a clean, generalizing win

Fixed RHF vs pre-fix on the same slice: entity recall +0.09, pair recall +0.03,
relF1@0.6 +0.064, and doc 3 went from 1 ent / 0 triples to 19 / 18. The held-out
slice confirms (entR 0.60 ≥ slice A's 0.56 — no overfit to the diagnostic docs).
No collapse occurred in any of the 25 runs. The fix is structural (coref can merge
but never delete), so this class of failure is closed, not patched.

## Verdict 2 — singlepass is still the recall champion, and it generalizes

Singlepass wins entity recall and pair recall on BOTH slices, does *better* on
held-out than on the diagnostic slice (entR 0.88), and costs 1 LLM call (~20–100 s)
per doc vs 14–24 calls (6–33 min) for the multi-stage strategies. Whatever the
final architecture is, this wide harvest belongs at the front of it.

## Verdict 3 — hybrid v1 landed in the middle; the funnel says exactly why

The wide front end works: it harvests ~the same pool as standalone singlepass
(e.g. doc 1: 30 vs 33; doc 4: 19 vs 20). The **deliberation back end then loses
15–30% of it**, per the stage funnel in `matrix_run.log`:

```
doc 1: 30 extracted -> 24 in KG      (coref kept 30; KG integration dropped 6)
doc 4: 19 extracted -> 15 post-coref -> 11 in KG
doc 30: Triples 13 extracted -> 10 approved -> 8 in KG
```

Three concrete leaks, in impact order:
1. **KG-integration drops** (stage 9): entities that survive extraction+coref
   vanish at integration — likely orphan pruning / governance thresholds. This is
   the single biggest recall leak in the fused pipeline. Needs an audit run with
   drop reasons logged.
2. **Harvested relationships discarded**: v1 stashes the wide call's relationship
   candidates (`wide_relation_candidates`) but nothing consumes them — the RHF
   relation stage rebuilds pairs from scratch and finds fewer (hybrid pairR 0.19
   vs singlepass 0.23 despite similar entity pools). v2 must seed the relation
   funnel with these candidates.
3. **Direction flips regressed** (7 on slice A vs 1 for both others) and entity
   precision dipped — coref canonicalization renames surfaces away from gold
   mention forms, and role-like entities ("coach", "producer") enter the KG.

## Verdict 4 — relation naming: RHF's style scores best when its pool is healthy

Fixed RHF posts the best relF1@0.6 on slice A (0.191) — deliberated, specific
relations match gold *meaning* well at the honest threshold. But its slice-B
relF1 (0.043) shows this doesn't yet generalize; small pair counts make this
metric noisy at n=5. Keep reading @0.6 only; @0.7/0.8 punish open-schema richness
by design (LESSONS.md Lesson 4) — an LLM-judge scoring pass is the planned
long-run answer.

## Recommendation

Keep the fused architecture — the bet is right, v1 just leaks:
1. **Audit + fix stage-9 KG-integration drops** (biggest, cheapest lever;
   log every dropped entity/triple with a reason, then decide thresholds).
2. **Seed the relation stage with `wide_relation_candidates`** so hybrid keeps
   singlepass's pair recall instead of rebuilding it.
3. Re-run this exact matrix (caches make it one command) — hybrid v2 should meet
   or beat singlepass on recall while adding deliberation's confidence/governance.
4. Then the deferred fixes: TIME/NUM value-entity decision (still the largest
   shared recall cap: gold's date/quantity nodes), direction post-check, and the
   Tier-1/Tier-2 self-improvement loop once the pipeline is stable.

Until hybrid v2 lands, **singlepass is the production default for recall-critical
ingestion; fixed RHF where governance/provenance matter most.**
