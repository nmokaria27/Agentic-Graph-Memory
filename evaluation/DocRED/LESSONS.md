# DocRED Phase 1 — Lessons (2026-07-03, docs 0–4, Re-DocRED dev)

Run: overnight_phase1.sh, RHF + singlepass on the same 5 docs, scored with
score_docred.py (embedding mode). Raw per-doc records with gold + text + calls in
`evaluation/results/docred_kg_cache/`.

## Headline numbers

| metric | RHF | singlepass |
|---|---|---|
| entity precision | 0.878 | 0.870 |
| entity recall | 0.472 | **0.809** |
| pair recall (either dir) | 0.143 | **0.232** |
| relation F1 @0.6 | 0.127 | **0.188** |
| wall / doc | 240–2280 s | 33–140 s |

Singlepass beats RHF on every aggregate here — the opposite of the synthetic-doc
verdict. The per-doc read shows *why*, and most of the gap is fixable bugs, not
strategy.

## Lesson 1 — CRITICAL: coref collapse reproduced WITHOUT self-consistency

`doc_3_rhf.json` ("Ramey Idriss"): 8 LLM calls, every one returned rich output
(2.2K–5.4K chars), zero empty/truncated — yet the final KG has **1 entity, 0
triples** (`ramey_idriss`). Extraction worked; consolidation destroyed it.
This is the same signature as the SC failure (29→1 via
`_stage4_coreference_resolution`), now on a plain RHF run. **It was never an SC
bug.** This single doc drags RHF entity recall from ~0.58 to 0.47 and pair recall
to 0. → Top fix: instrument stage-4 coref (log input entities + merge decisions),
find why unrelated entities merge into the protagonist, add a guard (e.g. never
merge entities with disjoint types / no shared mention token).

## Lesson 2 — RHF direction discipline is weak

RHF doc 0 emits flipped triples: `(medias_transylvania) -[BORN_IN]-> (schneider)`,
`(german) -[HAS_NATIONALITY]-> (schneider)`, `(gold_in) -[WON_MEDAL]-> (schneider)`.
The scorer counts flips only when the pair exists in gold (2 counted), but the
hand-read shows inversion is pervasive in relation-head-first output — the
relation-first prompt seems to lose track of which argument is the head.
→ Fix candidate: add an explicit "subject = the entity the sentence is about"
direction check in the relation-normalization stage, or a cheap post-pass.

## Lesson 3 — TIME/NUM/MISC gold entities dominate recall loss (expected)

Missed gold clusters by type, RHF: TIME 23, MISC 21, LOC 10, ORG 6, NUM 5, PER 3.
Gold treats every date/year/quantity as an entity with `start time`/`date of
birth` triples; our pipeline stores many of these as attribute values instead.
This is the value-vs-entity design decision flagged in PLAN.md — it caps pair
recall for BOTH strategies (singlepass misses them too, just fewer). Decide
once: either (a) emit date/quantity nodes when a fact links to them (singlepass
prompt already asks for this — that's part of its recall edge), or (b) accept
the cap and report DocRED scores with a "value-entities excluded" variant.

## Lesson 4 — relation-name style mismatch, not relation errors

RHF relations are verbose and specific (`SIGNED_CONTRACT_WITH_NATIONAL_TEAM`,
`PARTICIPATED_IN_OLYMPICS`) vs gold's generic Wikidata labels (`participant in`,
`employer`). At sim≥0.7/0.8 almost nothing matches (F1 0.028/0.015); at 0.6 many
do. The information is often *correct and richer* than gold — this is a scoring
artifact to keep in mind, and the 0.6 column is the honest one for open-schema
extraction. Don't "fix" richness to chase the 0.8 column (steer: quality first).

## Lesson 5 — RHF variance strikes again

RHF per-doc entity recall: 0.52 / 0.63 / 0.75 / 0.04 / 0.42 — huge spread.
Singlepass: 0.79 / 0.93 / 0.75 / 1.00 / 0.58. Even ignoring the doc-3 collapse,
RHF under-extracts on some docs (doc 4: 14 pred vs 26 gold ents). Confirms the
run-variance finding from EXTRACTION_EXPERIMENTS.md.

## Verdict for Phase 3

Priority order (fix → re-run SAME 5 docs, per PLAN.md doctrine):
1. Coref collapse (Lesson 1) — blocks trusting any RHF number.
2. Value-entity decision (Lesson 3) — biggest legitimate recall lever.
3. Direction check (Lesson 2).
Singlepass is the current DocRED baseline to beat; RHF's richness case (from the
synthetic experiments) doesn't show up here until 1–3 are fixed. A hybrid
(singlepass breadth + RHF deliberation for quality) remains the likely end state.
