# Direction-flip analysis — offline mining of v4/v5 caches (2026-07-06)

**Tool:** `analyze_flips.py` (reuses `score_docred.py`'s own matching — a "flip" here is
exactly the scorer's `flipped_only`). **Data:** slice A (docs 0–4) + held-out slice B
(docs 30–34), code state of matrix v4/v5. Zero LLM calls; run during Phase 4 legally.

## Raw counts (flipped-only pairs, 10 docs)

| strategy | flips | flips/doc | top gold relations |
|---|---|---|---|
| hybrid | 14 | 1.4 | country 7, located-in-admin 5, chairperson 4, creator 2 |
| rhf | 15 | 1.5 | country 5, located-in 5, creator 3, chairperson 3, birth/death dates 3 |
| singlepass | 4 | 0.4 | country 2, creator 1, chairperson 1 |

## The headline: most "flips" are NOT direction errors

Hand-classification of all 33 flip records (full list in
`evaluation/results/flip_analysis.json`, regenerable offline):

**Class A — correct inverse lexicalization (~55%).** The predicted triple is true and
directionally consistent with its own relation name; it just phrases the inverse of the
gold relation. `[george_tibbles --HEAD_WRITER_OF--> my_three_sons]` vs gold
`[show --creator--> person]`; `[prayut --LEADER_OF--> NCPO]` vs gold
`[org --chairperson--> person]`. All chairperson + creator flips in every strategy are
this class. **Not a system bug — a measurement gap**: L3 relF1 only scores
correctly-directed pairs, so true inverse phrasings are punished. The planned LLM-judge
scorer must judge semantic equivalence *including inversion* and report it separately.

**Class B — genuine inversions (~15%, rhf-concentrated).** The prediction violates its own
relation name's semantics: `[august_20 --BORN_ON_DATE--> ross_alger]`,
`[january_16 --DIED_ON_DATE--> ross]`, `[prelate_saskatchewan --BORN_IN_LOCATION--> ross]`,
`[university_of_toronto --EDUCATED_AT--> ross]` (the one hybrid case). All on doc 1
"Ross Alger" — a known pathological/reasoning-runaway doc. rhf has 4–5, hybrid 1,
singlepass 0. These are the real, fixable signal.

**Class C — entity containment-matching artifacts (~30%).** The predicted fact is fine but
one endpoint over-matched a gold cluster by substring containment:
`emperor_of_japan` ⊃ "japan" → matched the *Japan* cluster, so
`[emperor_of_japan --HOLDS_CEREMONY_AT--> kyoto_imperial_palace]` "flips" gold
`[palace --country--> Japan]`. Doc 33's entire country/located-in block (hybrid's top
counts!) and doc 31's `prime_minister_of_siam`→Thailand case are this class. Same
family as the known id-regex imperfection (ROADMAP §3.6): scorer containment matching
is a blunt instrument on multi-word titles that embed a place name.

## Implications

1. **`flipped_only` in MATRIX_REPORT overstates direction error ~3–6×.** Real inversions:
   hybrid ≈0.1/doc, rhf ≈0.45/doc, singlepass 0. Never treat raw flip counts as a
   direction-quality metric again; classify first.
2. **Hybrid's deliberation already suppresses most genuine inversions** (1 vs rhf's 4–5 on
   the same docs) — consistent with the v5 freeze decision.
3. The flip lever is SMALL. Post-check is worth having but will move pairR by ≤0.01;
   prioritize accordingly.

## Post-check design (implement AFTER Phase 4 — touches `multi_agent_kg/`)

1. **Value-subject rule (stage 9, zero LLM, domain-general).** A value-typed node
   (`classify_value` → DATE/TIME/NUMBER) must never be the SUBJECT of a triple whose
   object is a non-value entity — dates/quantities are attribute objects. Swap direction
   when violated. Fixes the BORN_ON_DATE/DIED_ON_DATE class with no relation vocabulary
   (no benchmark bias: uses only the system's own type predictions).
2. **Typed-argument spot-check (deliberation, bounded LLM).** For triples whose relation
   surface implies an argument order the endpoint types contradict (subject LOC/ORG/DATE +
   object PER on a `*_AT/*_IN/*_ON/*_OF`-shaped relation the system itself generated), ask
   one yes/no direction question. Expected volume: ≤1–2 triples/doc, only on pathological
   docs.
3. **Scorer-side (eval-only, can do anytime):** LLM-judge mode scores pred-vs-gold triple
   equivalence with inversion awareness; report `inverse_correct` as its own category so
   Class A stops polluting both flip counts and relF1.

Success bar for #1+#2 (pre-registered): rhf Class-B flips on docs 0–4 drop 4→≤1 with
entR/pairR/relF1 unchanged on slice B (structural fix, no benchmark tuning).

## Update (same day): #3 is implemented and smoke-validated

`score_docred.py --judge` (one batched Nemotron JSON call per doc, per-doc judge cache in
the kg-dir). Smoke on slice A singlepass: doc 3's known Class-A creator flip judged
`inverse` (`inverse_correct=1, genuine_inversions=0` — matches the hand classification
above), 0 judge errors. `precision_on_judged=0.654` vs embedding relF1 ≈0.1–0.2 range
confirms the embedding threshold under-credits relation meaning substantially. Full judge
pass over hybrid/rhf caches + Phase 4 caches: run AFTER Phase 4 (GPU contention only —
the mode itself is offline-on-cache).
