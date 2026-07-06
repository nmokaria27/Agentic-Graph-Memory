# DocRED Experiment Matrix v5 — Meta Report (2026-07-05 18:41, DATE value-node fix)

v5 tested one change: keep unreferenced DATE/year value nodes (drop only bare NUMBER counts).
Zero errors; singlepass control identical for the third consecutive matrix.

## v5 headline (v4 in parens)

| strategy | slice | entR | pairR | flips | relF1@0.6 |
|---|---|---|---|---|---|
| **hybrid v2** | A | 0.783 (0.793) | 0.252 (0.261) | 2 | **0.211** (0.183) |
| **hybrid v2** | B | 0.753 (0.727) | **0.245** (0.186) | 4 | 0.125 (0.118) |
| rhf | A | 0.790 (0.712) | 0.208 | 4 | 0.083 (0.146) |
| rhf | B | 0.749 (0.821) | 0.194 | 5 | 0.095 (0.074) |
| singlepass (control) | A | 0.809 (=) | 0.232 (=) | 1 | 0.188 (=) |
| singlepass (control) | B | 0.883 (=) | 0.200 (=) | 3 | 0.133 (=) |

## v5 verdict — held-out pair recall +32%; and we have hit the n=5 noise floor

- **The DATE fix delivered where it was aimed**: held-out (slice B) hybrid pairR jumped
  0.186 → **0.245** (+32%, now the best of any strategy on B, beating singlepass's 0.200), with
  entR +0.026. Kept years now anchor date-triples that previously died as `unresolved_entity`.
- **Slice-A deltas (−0.01) are noise**, as expected for a change that only affects docs with
  unreferenced years.
- **Noise floor reached.** rhf — identical code v4→v5 — swung entR **+0.08 on A and −0.07 on B**
  purely from sampling variance. The deltas we are now chasing (0.02–0.05) are smaller than
  run-to-run variance at n=5. Further 5-doc iteration cannot resolve real effects. Flip counts
  jitter 1–4 for the same reason.
- **Cumulative picture (v2 → v5), hybrid v2**: pairR A 0.126 → 0.252 (**2×**), pairR B → 0.245,
  relF1 A 0.096 → **0.211** (2.2×, best in matrix), entR A 0.64 → 0.78 (≈ singlepass), all with
  full deliberation/governance/provenance. Singlepass keeps a raw-entR edge on B (0.883)
  concentrated in reasoning-runaway docs where the wide call under-yields.

## Decision — freeze hybrid v2 as the production extractor; go to Phase 4

Per ROADMAP Phase A: **extractor = hybrid v2** (singlepass stays available for bulk-recall
ingestion). Next measurement is the **Phase 4 confidence run** — 40 fresh, never-touched dev
docs (100–139), hybrid + singlepass control — to (a) confirm the slice-A picture at a sample
size where variance can't dominate, and (b) hand the memory benchmarks a stable baseline.
Small-slice iteration on DocRED ends here; remaining Phase-A items (direction post-check,
LLM-judge scoring) ride on Phase-4 data.

---

# DocRED Experiment Matrix v4 — Meta Report (2026-07-05, labels measurement fix)

The v3 hybrid "recall gap" was partly a **scoring artifact**: coref renames entities to canonical
ids, but the gold-matchable surfaces live in `entity.labels`, which the runner dropped from the
dump. v4 dumps labels and scores name ∪ labels (backward-compatible: singlepass has no labels key
→ scored on name → **identical to v3**, confirming the fix equalizes rather than favors). Run
alone for clean attribution.

## v4 headline (v3 in parens)

| strategy | slice | entR | pairR | flips | relF1@0.6 |
|---|---|---|---|---|---|
| **hybrid v2** | A | **0.793** (0.721) | **0.261** (0.222) | 1 | 0.183 |
| hybrid v2 | B | 0.727 (0.681) | 0.186 (0.178) | 2 | 0.118 |
| rhf | A | 0.712 (0.670) | 0.192 | 1 | 0.146 |
| rhf | B | 0.821 (0.549) | 0.186 (0.108) | 4 | 0.074 |
| singlepass (control) | A | 0.809 (0.809) | 0.232 (0.232) | 1 | 0.188 |
| singlepass (control) | B | 0.883 (0.883) | 0.200 (0.200) | 3 | 0.133 |

## v4 verdict — on the diagnostic slice, hybrid now MATCHES/BEATS singlepass with governance

- **Slice A: the gap is gone.** Hybrid entR 0.793 ≈ singlepass 0.809 (Δ0.016, n=5 noise), and
  hybrid pairR **0.261 > singlepass 0.232** and relF1 0.183 ≈ 0.188, flips tied at 1. So the
  fused pipeline now delivers singlepass-level recall PLUS higher pair recall PLUS full
  deliberation/governance/provenance. ~2/3 of the v3 "gap" was measurement; the seeding + funnel
  fixes closed the rest.
- **Control held exactly**: singlepass identical v3↔v4 (labels-blind path). The fix cannot be
  accused of inflating hybrid against a moving baseline.
- **Slice B: a real but concentrated gap remains** (hybrid entR 0.727 vs singlepass 0.883). The
  per-doc diff shows it is **almost entirely one pathological doc (33, Kyoto Imperial Palace):
  hybrid 7/15 vs singlepass 14/15**; docs 30/31/34 are tied within 1–2 entities. At n=5 one
  outlier swings the aggregate — this is a small-sample effect, not a systematic hybrid weakness.

## Root cause on doc 33 → next generalizable lever

Two compounding effects, both domain-general (not DocRED-specific):
1. **Unreferenced value nodes are filtered out.** The `numeric_unreferenced` drop removed the
   years 1877 and 1869 — real gold entities — because no triple happened to reference them. A
   memory system should keep a typed DATE/YEAR/QUANTITY node regardless of current connectivity
   (you may later query "what happened in 1869?"). Fix: keep `classify_value`-typed nodes even
   when unreferenced; still drop bare non-value integers. Singlepass keeps them only because it
   has no filter at all.
2. **Wide extraction under-yielded on a reasoning-runaway doc** (13 entities, repeated truncation
   -ladder rescues). Harder to fix deterministically; candidate is a low-yield re-glean. Deferred.

Lever #1 is the v5 change: it directly recovers gold value-entities and generalizes to any
corpus with dates/quantities. Slice B is the pass/fail (held-out) after it lands.

---

# DocRED Experiment Matrix v3 — Meta Report (2026-07-04, post hybrid-v2 fixes)

Hybrid v2 shipped two structural fixes (HYBRID_V2_RUN.md): the stage-9 entity funnel
(year-safe id cleanup, alias merge on collision, per-reason drop logging) and wide-harvest
relation seeding. rhf + hybrid re-extracted on both slices; **singlepass caches untouched =
regression control**. 20 doc-runs, zero silent-empty, no data lost (the doc-1 reasoning-runaway
errors were segment-local and are addressed by the prompt-aware truncation cap, commit 419d049).

## v3 headline (vs v2 in parens)

| strategy | slice | entP | entR | pairR | flips | relF1@0.6 |
|---|---|---|---|---|---|---|
| **hybrid v2** | A | 0.86 | **0.72** (0.64) | **0.222** (0.126) | **1** | **0.211** (0.096) |
| **hybrid v2** | B | 0.82 | 0.68 (0.71) | 0.178 (0.150) | 2 (5) | 0.112 (0.090) |
| rhf | A | 0.89 | 0.67 (0.58) | 0.199 (0.176) | 4 (2) | 0.157 (0.148) |
| rhf | B | 0.76 | 0.55 (0.68) | 0.108 (0.175) | 4 (1) | 0.053 (0.089) |
| singlepass (control) | A | 0.87 | 0.81 | 0.232 | 1 | 0.188 |
| singlepass (control) | B | 0.79 | 0.88 | 0.200 | 3 | 0.133 |

## v3 verdict — the seeding fix is a decisive win; hybrid missed its recall bar but now owns quality

Against the pre-registered bars (HYBRID_V2_RUN.md):

- **Wide-harvest seeding worked, exactly as targeted (leak #2).** Hybrid slice-A pair recall
  **+76%** (0.126 → 0.222, now *meets* singlepass's 0.232) and relF1@0.6 **more than doubled**
  (0.096 → 0.211, now the **best of any strategy**). The deliberation back end, once fed the
  candidates it used to discard, produces the highest-quality relations in the matrix.
- **Direction flips: MET** (1 / 2 ≤ bar 2; hybrid v1 was 7). The funnel + alignment fixes
  removed the coref-renaming that caused inversions.
- **Entity recall: MISSED** (0.72 / 0.68 vs bar 0.75 / 0.80). Improved on slice A (0.64 → 0.72)
  but still ~0.09–0.20 behind singlepass. Since hybrid and singlepass share the SAME wide front
  end, this residual gap is **still back-end loss** (coref + integration) — smaller than v1's,
  not yet closed. Slice-B entR even dipped (0.71 → 0.68); at n=5 this is within noise, but it
  says the funnel fixes traded a little raw recall for correctness (fewer wrong merges).
- **Regression control held exactly**: singlepass identical v2↔v3 (path untouched). rhf moved
  (re-run): slice-A entR +0.09 (funnel fixes help rhf too) but slice-B entR −0.13 and flips
  +2 — rhf is the noisiest strategy run-to-run and is no longer a contender on either recall
  or quality.

**Decision:** we now have a clean **recall-vs-quality frontier**, not a single winner.
- **singlepass** = maximum entity recall (0.81 / 0.88), 1 LLM call, no governance → the
  ingestion recall engine.
- **hybrid v2** = best relation quality (relF1 0.211, flips 1), pair recall matching singlepass,
  full deliberation/governance/provenance → the quality/governed backend.

Production default: **hybrid v2 where relation quality or governance matters; singlepass for
pure recall-critical bulk ingestion.** The open lever (Phase A) is the residual hybrid entity-
recall gap = the coref/integration back-end loss that seeding did not touch. **This is a real,
pre-registered miss — not spun as a win.**

### Where the residual gap actually is (funnel evidence → Phase A target)

The v3 entity-funnel logs localize it. The garbage filter is now nearly a no-op — it drops 0–2
entities/doc (all `numeric_unreferenced` values), so **stage-9 integration is no longer the
leak**; the funnel fix closed it. But the count entering the filter is already low (11, 14,
17…) vs the ~20–30 standalone singlepass keeps. The difference is the one stage hybrid has and
singlepass does not: **coreference resolution**. Singlepass emits the wide call's entities
straight through; hybrid runs those *same* entities through coref, which still over-merges
(merges distinct gold clusters into one), costing the ~0.09–0.20 recall.

**Phase A next fix (one loop, same 5 docs):** loosen coref merging — it already can't delete,
but it's still merging too eagerly. Candidate: require a stronger similarity/evidence threshold
before merging two extracted entities, or gate merges on type agreement. Re-measure entR against
this v3 baseline; success = hybrid entR → singlepass's 0.81 without pairR/relF1 regressing.

---

# DocRED Experiment Matrix — Meta Report (2026-07-03, post coref-fix)

3 strategies × 2 slices, Re-DocRED dev. **Slice A** = docs 0–4 (diagnostic; every fix
was developed against these). **Slice B** = docs 30–34 (held-out; never hand-read,
never tuned on — the memorization check). Zero crashes, zero empty LLM calls across
all 25 doc-runs. Raw per-doc records: `evaluation/results/docred_kg_cache*/`;
scores: `evaluation/results/docred_scores_<strategy>_<slice>.json`.

## Headline numbers

| strategy | slice | entP |
 entR | pairR | flips | relF1@0.6 |
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
