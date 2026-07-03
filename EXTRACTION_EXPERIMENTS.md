# Extraction-Strategy Experiments

Goal: find the best entity/relation extraction backend for this system on **local Nemotron 30B**
(vLLM, gpu02), **without** losing the system's differentiators — domain-adaptivity (the graph
must reshape its schema for law / medical / any incoming domain), governance, and conflict
resolution. We experiment and measure before committing.

Constraint from the project owner: **no fixed general-purpose extractor models (e.g. GLiNER)** —
the schema must be discovered per-corpus by `DomainClassifier` so the system self-organizes.

Harness: `scripts/extraction_experiment.py` (same document, same model, instruments every LLM
call). Metrics: wall-time, #LLM calls, entities, triples, orphan rate, empty-output calls.

## Environment
- Model: `nvidia/nemotron-3-nano` → `/scratch/models/nemotron-nano-30b-fp8` (FP8, 131K ctx), vLLM on gpu02.
- Embeddings: `mxbai-embed-large:335m` (gpu01 Ollama).
- Nemotron is reasoning-capable; `NEMOTRON_THINKING` env toggles it (default `on`).

## Key finding that motivates these experiments (profiling the current RHF pipeline)
A single small segment through the full 9-stage pipeline took **1107 s** (15-fact doc) / **630 s**
(10-fact doc). Breakdown of the 630 s run (22 LLM calls, 99% of wall time in the LLM):

| stage bucket | calls | sum s | note |
|---|---:|---:|---|
| gleaning ("recover MISSING triples") | 4 | **297 (47%)** | 3 of 4 returned **empty** after ~74 s each |
| evidence linking | 1 | 62 | 11.5 KB output |
| entity extraction | 2 | 47 | |
| domain classify | 1 | 36 | |
| verification | 1 | 32 | |
| relation identification/pairwise | ~5 | ~90 | |

Root cause: Nemotron spends the (4×-inflated) token budget on hidden reasoning; on the biggest
prompts it truncates before emitting JSON → **74 s wasted producing nothing**. This — not RHF's
logic — is why benchmark contexts time out / crash mid-run and leave near-empty KGs.

`NEMOTRON_THINKING=off` cut the same run **630 s → 150 s (4.2×)** — but the current *verbose* RHF
prompts make Nemotron reason past a 4096 budget even with reasoning off, truncating to **0
entities**. Lesson: the fast path needs **short, single-pass prompts**, which is exactly the
GraphRAG-style experiment below. (Default left at `on` so nothing is broken meanwhile.)

## Strategies under test
- **rhf** — current `DeliberativeOrchestrator` multi-stage RHF (Relation-Head-First): identify
  relation types → bind heads → bind tails → glean. Reasoning ON. Baseline.
- **singlepass** — GraphRAG-style: ONE call extracts entities+relationships together, guided by
  the domain types discovered by `DomainClassifier` (open-world: may invent new types → stays
  adaptive), reasoning OFF, + 1 gleaning round to connect orphans.

## Results (12-fact mixed-domain sample)

| strategy | reasoning | domain (auto) | wall s | LLM calls | entities | triples | orphan rate | empty calls |
|---|---|---|---:|---:|---:|---:|---:|---:|
| singlepass | **off** | TECH_BIOTECH_CASE_NARRATIVE | 172 | 7 | **0** | 0 | – | **3** |
| singlepass | **on** | TechBiotech Innovation Landscape | **198** | **6** | **20** | **15** | **0.0** | 0 |
| rhf (baseline) | on | (same doc) | 795 | 25 | 22 | 14 | **0.136 (3 orphans)** | **4** |

**Count-only verdict (misleading):** single-pass looked *≥* RHF — more triples (15 vs 14), zero
orphans (vs 3), zero empty calls (vs 4), at 1/4 the wall time and 1/4 the LLM calls.

**Qualitative (graph-dump) verdict — RHF actually produces the RICHER, more CORRECT graph.**
Reading the two dumps (`exp_singlepass_kg.json`, `exp_rhf_kg.json`) fact-by-fact against the
12-sentence source overturns the count-only read. The extra single-pass triple came *bundled with*
a semantic error and a dropped entity:

| fact / property | single-pass | RHF | winner |
|---|---|---|---|
| Meridian Bridge ↔ Ashford | `LOCATED_IN` (source says *completed BY*) | `INFRASTRUCTURE_COMPLETED_BY_ENTITY` | **RHF (SP wrong)** |
| Trellis University (Elena's lab) | **entity dropped entirely** | `elena WORKS_AT trellis_university` | **RHF** |
| Marcus's CEO role | dropped | `ceo` node + `STEPS_DOWN_FROM_POSITION` | **RHF** |
| vehicle colors | "Toyota" / "Honda" (color lost) | `blue_toyota` / `red_honda` | **RHF** |
| relation vocabulary | generic (`DISCOVERED`, `MANAGES`) | typed (`SCIENTIST_DISCOVERS_SUBSTANCE`) | **RHF** |
| enzyme typing | `Kessler enzyme::PRODUCT` (wrong) | typed internally | **RHF** |
| fact 12: enzyme → hydrothermal vents | **captured** | **missed** | **single-pass** |
| orphans | 0 | 3 (governance-*inactive*, not a recall gap) | tie |
| cost | 6 calls / 198 s | 25 calls / 795 s, 4 empty | **single-pass** |

RHF's *only* genuine recall miss (fact 12, enzyme→vents) is **not a design flaw — it is the
empty-gleaning-truncation bug** profiled above: the gleaning stage that recovers MISSING triples
returned empty 3/4 times because Nemotron spent its budget on reasoning and truncated before JSON.
**Fix gleaning and RHF captures fact 12 too, dominating on richness outright.** Single-pass's zero
orphans / low cost are real, but its graph is *thinner and locally wrong* (Meridian semantics,
dropped Trellis, lost roles/colors/dates) — which fails the "rich, solid and good" bar.

RHF profiling references (other docs, reasoning on): 10-fact → 22 ent / 17 tri / 630 s / 22 calls;
15-fact → 29 ent / 18 tri / 1107 s.

## Findings so far

1. **Reasoning-OFF is a dead end for extraction on this Nemotron build.** Even a short single-pass
   prompt returns **empty** (0 entities, 3 empty calls) — the model needs its reasoning channel to
   produce extraction output; "detailed thinking off" just yields blank content. So speed cannot
   come from disabling reasoning. (Kept `NEMOTRON_THINKING=on` as default.)

2. **Single-pass is fast but thinner and locally wrong — NOT the quality win.** The graph dump
   shows it drops entities (Trellis University), loses attributes (vehicle colors, roles, dates),
   mistypes (`Kessler enzyme::PRODUCT`), and inverts one relation (`Meridian LOCATED_IN Ashford`
   vs the correct *completed by*). It is a good **cheap high-recall first pass**, not the
   final-quality backend. RHF's typed relations + role/attribute nodes are richer and more correct.

2b. **RHF is the richness winner once gleaning is fixed.** Its lone recall miss (fact 12) is the
   truncation bug, not its design. Its 4 empty reasoning calls are the same wasted-budget symptom.
   Both are fixable without abandoning RHF.

3. **DomainClassifier is a hidden ~90 s tax.** It runs self-consistency (3+ calls) + a low-conf
   escalation on every build — roughly half of single-pass wall time. Gating self-consistency
   (or caching the domain per corpus) is an independent speed lever worth ~2× more on single-pass.

## Interpretation & decision (revised after reading the graph dumps)
The count-only read favored single-pass; the **qualitative** read reverses it. Per the owner's
priority — *"rich, solid and good"* data over speed — **RHF produces the better graph** (correct
Meridian semantics, kept Trellis/CEO/colors, specific typed relations). Single-pass is thinner and
has a semantic error, so it is **not** the quality backend; it is a strong *cheap recall* pass.

**Decision — fix RHF rather than replace it, then optionally hybridize:**
1. **Fix the empty-gleaning truncation** (the root cause of both RHF's fact-12 miss AND the 4 empty
   calls / 297 s waste): give gleaning/large prompts enough completion budget *after* reasoning, or
   split the reasoning and answer turns so JSON is never truncated. This is the single highest-value
   fix — it recovers real recall (fact 12) and reclaims ~47% of wall time simultaneously.
2. **Keep RHF as the primary extractor** (richness backend), reasoning ON, domain-adaptive schema
   intact.
3. **Optional hybrid:** run single-pass first as a cheap high-recall seed, then RHF's head/tail
   binding + gleaning to enrich typing, roles, and attributes and to connect what single-pass
   dropped. Single-pass's fact-12 hit + RHF's Trellis/CEO/color hits are *complementary* — a union
   would be strictly richer than either alone.
4. **Cache the domain classification per corpus** (independent ~90 s/build lever; does not affect
   quality).

Next action: implement fix #1 (gleaning budget) and re-run the same-doc harness to confirm RHF now
also captures fact 12 with zero empty calls, before wiring anything into the benchmark adapter.

## Validation of fix #1 (truncation-retry in `chat_completion`)

Fix applied in `multi_agent_kg/llm/openai_client.py`: when a thinking model returns EMPTY content
with `finish_reason="length"` (budget consumed by hidden reasoning), the client now doubles
`max_tokens` (capped at `VLLM_MAX_MODEL_LEN − 2048`) and retries inside the existing retry loop,
instead of silently returning `""`. Loud WARNING if the cap is hit. Unit test:
`test_truncated_thinking_output_grows_budget_and_retries`. Full suite: 206 passed.

Same-doc RHF re-run (fix live):

| metric | RHF before fix | RHF after fix |
|---|---:|---:|
| empty calls | **4** | **0** |
| gleaned triples added | 0 | **5** |
| fact 12 (enzyme → hydrothermal vents) | missed | **captured** (`SUBSTANCE_FOUND_IN`) |
| Meridian semantics | correct | correct (`PROJECT_COMPLETED_BY`) |
| wall s / LLM calls | 795 / 25 | 574 / 17 |
| entities / triples | 22 / 14 | 20 / 11 |

Observed live in the log: `truncated before output (finish_reason=length) at 16384 tokens;
raising budget -> 32768 and retrying` — followed by a non-empty gleaning result. The mechanism
that made benchmark contexts collapse to near-empty KGs is closed.

**Caveat surfaced by the re-run: run-to-run variance is now the biggest quality risk.** This run
*gained* fact 12 and kept Trellis/typed relations, but *dropped* facts the baseline run had: Bob's
relocation to Seattle, both vehicles (blue Toyota / red Honda), and Marcus's CEO step-down — and
drifted `marcus_webb` → `marcus_webber`. Neither run is a superset; the union of the two RHF runs
covers all 12 facts. This is temperature-sampling variance in the upstream entity/relation
identification stages, not the gleaning fix. Levers, in order of promise given the owner's
quality-over-speed priority:
1. **Enable self-consistency in the orchestrator** (`enable_self_consistency=True`, disabled in
   this harness): multiple samples + agreement voting directly attacks variance; costs more calls,
   which is acceptable.
2. Lower extraction temperature toward 0 for the identification stages.
3. Hybrid union (single-pass seed ∪ RHF enrich) — complementary misses observed empirically.

## Self-consistency: consensus rewrite + first live run (FAILED, instructively)

The stock SC was confirmed broken by inspection: `_compute_consensus` voted on the EXACT
serialized whole response; at temperature 0.7 multi-item JSON never matches byte-for-byte, so SC
returned an arbitrary sample at 1/n confidence — 3× cost, zero variance reduction (this also
explains DomainClassifier's perpetual low-confidence escalation ≈ the 90 s tax). Rewrote it as
**item-level consensus** in `agents/base.py`: pool items across samples, dedupe by identity
fields (subject/relation/object/text/id — entity `type` deliberately excluded so PERSON vs
EMPLOYEE merges), volatile fields (confidence/rationale) excluded from identity, union kept
(not majority — observed variance means true facts often appear in 1/3 samples) with
cross-sample support blended into confidence (0.5 + 0.5·support, averaged with model
confidence). 7 unit tests (`tests/test_self_consistency.py`); suite: 213 passed.

**Live run (rhf --sc, same 12-fact doc): collapsed to 1 entity / 0 triples in 1458 s.** Two
distinct failures observed:

1. **Temp-0.7 reasoning runaway.** SC samples at temperature 0.7 send Nemotron's reasoning
   channel into rambling; the truncation-retry ladder escalated 16K→32K→65K→129K tokens on
   multiple calls (the retry FIX worked — 0 empty calls — but each rescue costs minutes).
   SC sampling temperature must come down (~0.4–0.5) and/or `call_llm_with_self_consistency`
   must pass an explicit `max_tokens`.
2. **Coref collapse: 29 → 1 entities.** Entity extraction + union consensus produced a healthy
   29 entities; `_stage4_coreference_resolution` (plain call_llm, NOT SC) then merged them into
   ONE entity (`meridian_bridge`). Hypothesis: the unioned entity list carries near-duplicate
   variants (same text, different sample-assigned ids — `id` is an identity field, so variants
   don't dedupe) and/or heterogeneous field shapes, and the coref model at that input emitted
   one giant group. NEEDS a controlled repro with coref I/O logged before SC is trusted.

Verdict: item-level consensus is correct at the unit level but **SC is not production-ready**;
it stays out of benchmark configs until (a) SC temperature/budget tuned, (b) union keys entities
by text (not sample-local id), (c) coref collapse root-caused. Tracked as the top open item.
