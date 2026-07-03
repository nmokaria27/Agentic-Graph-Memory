# DocRED (Re-DocRED) Evaluation Plan

Goal: use Re-DocRED as a **diagnostic instrument first, scoreboard second**. Small slices →
lessons → incremental system fixes → re-measure on the SAME slice → only then scale up.
This replaces eyeballed 12-fact toy docs with **gold triples** (objective P/R/F1).

## Why Re-DocRED (not original DocRED)
Original DocRED is missing ~60% of true triples (false negatives) — an open-world extractor
would be *punished* for correct extractions. Re-DocRED (Tan et al., EMNLP'22) revises the
labels. Data lives in `evaluation/DocRED/data/`:
- `dev_revised.json` — 500 docs (our eval split; test labels are for final runs only)
- `train_revised.json` — 3,053 docs (source of few-shot examples if ever needed; never eval)
- `rel_info.json` — 95 P-code → English name map (built from Wikidata API)

## Dataset shape (measured)
- Avg per dev doc: **19.4 entities, 34.6 gold triples**, ~6 sentences (~900 chars → 1 segment)
- Entity = cluster of mentions with aliases + type (PER/ORG/LOC/TIME/NUM/MISC)
- Gold triple = (head_idx, tail_idx, P-code) + evidence sentence ids
- Gold density ≈ **1.8 triples/entity** — a concrete richness bar (our toy-doc runs: ~1.2)

## Scoring design (`score_docred.py`)
Predicted triples are open-world (free-form names); gold uses entity clusters + 95 relations.
Matching is deliberately *soft-to-strict*, reported at every level so we see WHERE loss happens:

1. **Entity match**: normalized predicted surface (lower, strip punct/underscores) matches ANY
   mention alias in a gold cluster. Report entity P/R/F1 first — if entities don't land,
   nothing downstream can.
2. **Pair recall**: predicted (head, tail) hits a gold pair in either direction, ignoring
   relation. Isolates "did we even connect the right things".
3. **Relation match** (two modes):
   - `--mode open`: predicted relation name ≈ gold relation name via embedding similarity
     (mxbai, threshold sweep 0.60/0.70/0.80) — evaluates the system AS DESIGNED (adaptive).
   - `--mode fixed`: pipeline runs with `schema_override` = the 95 gold relations +
     6 entity types (`enable_open_world=False`) — exact match, comparable to published
     DocRE numbers (loosely; we're generative, they're classifiers).
4. Report micro P/R/F1 at each level + per-relation confusion for the top misses.

Direction: gold triples are directed; count reversed direction separately (a *lesson* category,
not silently wrong or silently right).

## Runner design (`run_eval.py`, mirrors LoComo/MAB conventions)
- `--max-docs N --offset K` — fixed, seeded slice so re-runs after fixes are comparable
- `--strategy rhf|singlepass` + `--sc` — same knobs as `scripts/extraction_experiment.py`
- `--save-kg-dir evaluation/results/docred_kg_cache/` — **checkpoint per doc**: KG JSON +
  per-call timing dumped after EACH doc, so a crash loses ≤1 doc (robustness lesson from the
  "122→9" era) and scoring can re-run offline without touching the LLM
- `python -u` + tee to log; instrument LLM calls (count, wall, empty, truncation-retries)
- Score step is a separate offline command reading the cache — never re-extract to re-score

## Phases & gates
**Phase 0 — smoke (1 doc, ~10 min).** Doc #0 through RHF. Gate: pipeline completes, KG dumped,
scorer runs, zero silent-empty LLM calls. *Watch: DomainClassifier verdict on Wikipedia prose;
segment count; any truncation-retry warnings.*

**Phase 1 — diagnostic slice (5 docs, ~1 h).** Docs 0–4, RHF (current best config). Read every
graph against gold by hand. Deliverables: entity/pair/relation P/R/F1 + a written top-5 failure
mode list (e.g., TIME/NUM values dropped, direction flips, granularity mismatch, alias misses).
Gate: we can NAME why each lost triple was lost.

**Phase 2 — strategy matrix (same 5 docs).** RHF vs singlepass vs (SC once fixed) on the
identical slice. Also `--mode fixed` vs `--mode open`. This is EXTRACTION_EXPERIMENTS.md's
table with gold instead of my judgment.

**Phase 3 — fix → re-measure loop.** Each failure mode becomes one incremental change
(prompt, alias handling, direction hint, value-entity handling). Re-run the SAME 5 docs after
each change; accept if F1 ↑ and nothing regresses. Log every accepted/rejected change in
`evaluation/DocRED/LESSONS.md`.

**Phase 4 — confidence run (30–50 docs, overnight).** Best config from Phase 3. Compare
open-mode F1 trend vs Phase 1 to check the 5-doc slice wasn't lucky. Only after this do we
consider the 500-doc dev set.

Then: LongMemEval smoke (its per-ability splits — esp. knowledge-updates — test supersede,
which DocRED cannot). Same doctrine: 5-question slice per ability first.

## Known risks / open items
- Nemotron reasoning runaway at high temperature (seen in SC validation: budget ladder to 129K,
  24-min wall) — SC needs temp/budget tuning before it enters the matrix.
- Coref collapse under SC (29 entities → 1) — MUST be root-caused before SC is trusted anywhere.
- TIME/NUM gold entities ("13 March 1963", "2002") — our pipeline may treat these as attribute
  values, not entities; expect this to dominate early recall loss. Decide: value-nodes vs
  attributes (this is a *design decision* DocRED will force, and it matters for the KG's shape).
- Wikipedia prose ≠ conversational memory — DocRED validates extraction, NOT the memory system;
  don't over-fit the pipeline to it. LongMemEval covers the other half.

## Cost model (measured baselines)
RHF ≈ 10 min/doc, ~17 LLM calls → 5 docs ≈ 1 h; 50 docs ≈ overnight. Singlepass ≈ 3 min/doc.
SC currently ~24 min/doc (broken — see risks). All local (gpu02 vLLM), zero API cost.
