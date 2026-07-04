# Hybrid v2 Run — definition & protocol

**Goal:** close the two back-end leaks that kept hybrid v1 below standalone singlepass
(MATRIX_REPORT.md verdict 3), then re-measure the full matrix. Hybrid v2's success bar:
**meet or beat singlepass entity/pair recall while keeping deliberation's confidence,
governance and provenance.** If it can't, the fused architecture gets rethought — measured,
not assumed.

## Changes under test (all structural, none DocRED-specific)

1. **Stage-9 entity funnel: audit + fix (knowledge_organizer.py `_integrate_to_kg`)**
   - `(?<=\w{3})_\d+$` id-cleanup stripped ANY trailing digits — collapsing year-suffixed ids
     (`fibt_world_championships_1998` → `fibt_world_championships`), silently merging distinct
     entities. Now strips only 1–2 digit artifact counters (`entity_2`), never 4-digit years.
   - Same-id collisions were skipped WITHOUT merging labels — the second entity's aliases
     (gold-matchable mention forms) were discarded. Now every entity flows through
     `add_entity`, which merges labels; collisions are counted, not silenced.
   - Every drop now has a counted reason (`empty_text / numeric_unreferenced / garbage_phrase /
     too_short / garbage_type`), printed as an entity-funnel line mirroring the existing
     triple-funnel — no more "6 entities vanished somewhere."

2. **Wide-harvest relation seeding (relation_extractor.py + deliberative_orchestrator.py)**
   - The wide front end's relationship candidates (`wide_relation_candidates`) were stashed and
     never consumed; RHF rebuilt pairs from scratch and found fewer (hybrid pairR 0.19 vs
     singlepass 0.23). Now the orchestrator passes them into `RelationExtractor.run`, which
     aligns them to the entity catalog, dedupes against RHF-extracted triples, tags them
     `seeded_from_wide`, and reports `seeded_triples_added` in the funnel.

## Run protocol (matrix v3)

- Same two slices as v1/v2 for comparability: slice A = dev docs 0–4 (diagnostic),
  slice B = dev docs 30–34 (held-out).
- Strategies re-run: **hybrid + rhf** (both traverse the changed organizer/relation code).
  Singlepass caches remain valid (its path is untouched) — it stays the benchmark to beat.
- Stale `doc_*_rhf.json` / `doc_*_hybrid.json` caches are DELETED first (pipeline changed;
  cache-skip would silently reuse v2 outputs).
- Score all 3 strategies × 2 slices with the layered scorer (entity → pair → relation@0.6/0.7/0.8,
  direction flips, missed-type breakdown) → `docred_scores_v3_*`.
- Script: `evaluation/DocRED/matrix_v3.sh`, detached, per-doc checkpoints, self-logging to
  `evaluation/results/matrix_v3.log`. Monitored live; course-corrections logged in this file.

## Duration estimate

20 pipeline docs (hybrid ×10 ≈ 10–20 min/doc, rhf ×10 ≈ 6–20 min/doc) ≈ **4–7 h** end-to-end
(v2 measured in this band), plus ~10 min scoring. Zero API cost (local vLLM gpu02 + Ollama gpu01).

## Success criteria (read tomorrow morning)

| metric | hybrid v1 | bar for v2 |
|---|---|---|
| entity recall (A / B) | 0.61 / 0.66 | ≥ 0.75 / ≥ 0.80 (near singlepass 0.81/0.88) |
| pair recall (A / B) | 0.19 / — | ≥ 0.23 / ≥ 0.20 (meet singlepass) |
| direction flips (A) | 7 | ≤ 2 (back to rhf/singlepass level) |
| entity funnel | opaque | every drop has a printed reason |
| regression guard | — | rhf + singlepass scores within noise of v2 |

## Overfitting guard (why this isn't "tuning for DocRED")

Every change is a structural bug-fix visible without gold labels (years eaten by a regex,
aliases discarded on collision, harvested candidates never consumed). The held-out slice B
catches slice-A tuning; singlepass acts as an untouched control; and the next benchmark
(LongMemEval — conversational, temporal, knowledge-update) shares none of DocRED's
distribution, so fixes that only help DocRED will show up there as no-ops or regressions
and get reverted. DocRED-specific *policies* (e.g. value-entity nodes) are recorded as
explicit design decisions in LESSONS.md, not silently baked in.
