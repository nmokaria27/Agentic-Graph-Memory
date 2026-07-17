# Lessons & Experiments — the Agentic Graph Memory improvement loop

*A knowledge-transfer document. Written 2026-07-14, after ~20 pre-registered
experiments in ~8 days transformed this system. Intended as a learning/inspiration
doc for adjacent projects in the graphs + memory domain. Everything here is backed
by entries in `EXPERIMENT_LOG.md` (the source of truth, with pre-registrations,
bars, and honest verdicts) — this doc is the narrative and the distilled lessons.*

---

## 1. What we were building

A **governed knowledge-graph memory system** for agents. The one-sentence thesis:
*extraction is not memory* — a pile of extracted triples is a flat store; memory
requires knowing who owns a fact, why it was admitted, what superseded it, and
how to route a question to the right subgraph.

Concretely, a Governed Knowledge Graph `G = (E, T, D, φ, γ)`: entities, triples,
governed **domains** discovered per-corpus (no fixed schema), an ownership map
`φ`, and a governance router `γ` that sends each proposed triple to a domain
owner for approve/reject/revise/escalate. Provenance (including **source-document
dates**) rides on every triple, so a conflict resolver can supersede stale facts
by recency. QA routes through domain-owned subgraphs.

The standing requirements the owner never let us trade away:

1. **Robust** — no silent mid-run data loss, ever. Partial results preserved, drops counted and printed.
2. **Domain-adaptive** — schema discovered per corpus; a fixed general-purpose extractor (GLiNER etc.) was explicitly rejected.
3. **Quality first** — rich correct KG data beats speed.
4. **No benchmark bias** — improvements must be structural/domain-general; the system must excel on domain data it sees *without memorizing benchmarks*.
5. **Experiment → document → decide** — pre-register, measure against a bar, verdict honestly.
6. **Production inference is local** (30B-class model on 2× L40S); API lanes are for parallel experimentation only.

## 2. The method (the most transferable part)

The improvement loop itself mattered more than any single fix:

- **Pre-registration before running.** Hypothesis, exact change, slice, control,
  and numeric bars are committed to git *before* the result exists. This killed
  score-chasing at the root: a change phrased as "makes the score go up" cannot
  be pre-registered because it names no mechanism.
- **One structural change per experiment.** Every accepted change names the
  domain-general mechanism it fixes ("dates must survive ingestion", "distinct
  literals can never be the same entity").
- **Controls that cannot move by construction.** Every new capability ships
  default-off behind a flag/env; the control configuration is bit-identical.
  When a "measurement fix" changes the control's numbers, it's a bug or bias.
- **Dev/held-out splits on every benchmark.** Anything hand-read or debugged
  against is dev data *forever*. Held-out slices are scored at most once per
  accepted change and never inspected.
- **Honest verdicts, including REVERTs.** Roughly a third of our experiments reverted —
  and the reverts produced the most valuable findings (see §5.1).
- **Offline before online.** Cached per-doc artifacts + offline scorers meant we
  could kill bad hypotheses in minutes for $0 (see §5.4).
- **Two compute lanes.** Local GPU = production truth; a cheap API lane
  (env-routed, zero code change) ran experiments in parallel while the local lane
  was measuring. Rule: an API-lane win is *evidence*, not a result — reproduce
  locally before adopting.
- **Crash-safe runs**: `nohup` + done-markers in logs + per-item checkpoints, so
  reruns are free and a dead SSH session never loses a benchmark.

## 3. The system, as it ended up

Pipeline (per document): DocProcessor → DomainClassifier (+ governance bootstrap)
→ EntityExtractor (deliberative or wide mode; coref with merge-never-delete) →
RelationExtractor (or wide-harvest triples directly) → Connectivity pass
(disconnected entities) → Pair-completion pass (opt-in; co-occurring unlinked
pairs) → Evidence linking → Deliberation → Verification (batched, evidence-based)
→ KnowledgeOrganizer (guarded dedup → propose_triple → governance → commit with
provenance).

Load-bearing details that took experiments to get right:

- **Stage-9 dedup guards**: merge-into-aliases (a merged entity's surface forms
  are preserved on the canonical — nothing gold-matchable is ever deleted);
  merge-group size caps; type-compatibility gate; numeric/date **literal guard**.
- **Entity types must physically survive** extraction → coref → organizer, or
  every type gate silently no-ops (GB-11 — see §5.2).
- **Freshness**: `document_date` threads from ingestion metadata → AgentContext →
  provenance refs → the conflict-resolver prompt, giving supersede decisions a
  recency signal. Undated corpora see zero behavior change.
- **Parallel batch fan-out** (`map_batches`): verification/coref/evidence/
  connectivity batches are independent; a bounded thread pool with
  submission-order aggregation cut wall 38% (API lane) / 20% (local). Default-off.
- **Embedding failover**: primary endpoint exhausts its retry ladder → sticky
  process-level switch to an API embedding model, loud warning, run survives.

## 4. The complete experiment chronicle

### Era 0 — the paper (pre-loop baseline)
SciERC test 100 docs on GPT-5, fixed schema: flat insertion strict-F1 0.106 →
MAGG governed 0.156 (+47%), mapped 0.192 → 0.290 (+51%), human review preferred
governed triples, MuSiQue QA token-F1 0.646 vs 0.480 flat. Cost: ~8M GPT-5
tokens / 100 docs. This is what the loop had to beat *locally*.

### Era 1 — extraction matrix & the model question
- **Matrix v1–v5 + Phase 4 (n=40)**: the "obvious" architecture (hybrid: harvest
  front-end + full deliberative back-end) LOST entity recall on 29/40 docs vs
  plain singlepass at 30× the cost. The freeze on hybrid was **overturned by
  scale-up** — n=5 diagnostics had flattered it. **Singlepass became the
  production extractor.**
- **EXP-JUDGE-PHASE4**: an LLM-judge adjudication showed relation *naming* was
  NOT the problem (singlepass ≥ hybrid on every judge metric) — the recall gap
  was pair discovery. This redirected months of would-be prompt work.
- **EXP-HEADROOM-GLM / EXP-MODEL-LOCAL**: a bigger API model (+0.08–0.10 entR)
  proved the ceiling was partly model-bound; swapping the local model
  (Nemotron-30B → Qwen3-30B-A3B FP8) hit entR **0.908** at **8 s/doc** — a
  simultaneous quality and 10× speed win. Lesson: measure model headroom cheaply
  on an API proxy before buying local swap effort.

### Era 2 — robustness (the wipeout arc)
- **EXP-LMESMOKE-PRO/FLASH**: first LongMemEval smokes. 3 of 7 runs silently
  produced empty KGs (one bad stage output → whole document wiped). A JSON-shape
  crash in dedup cost entire documents. Verdict: requirement #1 violated — the
  smoke "earned its keep three times over."
- **EXP-ROBUST-DEGRADE (GB-1)**: graceful stage degradation — every stage failure
  degrades to passthrough with counted drops; Pydantic validation at every LLM
  boundary. **EXP-ROBUST-VALIDATE**: the twice-wiped question now commits 506
  entities. ACCEPT.
- Model quirks catalog (kept in the ops guide): one API model returned nested
  JSON fragments that silently parsed to 0 entities; another 500'd server-side;
  a reranker was mislabeled as an embedder. **Trust only what you tested through
  your own client.**

### Era 3 — governance & freshness (the "thesis gap")
- The KG stored updated facts but QA served stale ones ("$350k" vs "$400k") —
  confirmed 3×. Root cause was NOT the conflict detector: the resolver prompt
  simply had **no recency signal**.
- **EXP-FRESHNESS-QA (GB-2)**: thread document dates end-to-end into provenance
  and the resolver prompt. ACCEPT.
- **EXP-FRESHNESS-E2E (GB-2b)**: per-session dated ingestion → supersede events
  went from ~0 to 1–20 per question; first knowledge-update gate pass. Along the
  way found **GB-10**: the advanced QA orchestrator crashed in its fallback path
  (`AttributeError` on a method only the base class defined) — latent for weeks,
  detectable only under real load.
- **EXP-GB2C-RELATION-EMBED**: embedding-similar relation names as conflict
  candidates. Unit tests passed with a fake embedder; **real embeddings failed
  the motivating case** (0.69 similarity vs 0.85 bar — `has_time`/`has_value`
  aren't semantically close even when they name the same fact). PARTIAL — shipped
  for genuine synonyms, honestly recorded as not closing the target case.

### Era 4 — the SP-GOV arc (reverts that paid)
- **EXP-SPGOV (GB-9)**: "governed singlepass" — harvest → verify → govern,
  skipping the expensive middle. REVERT (entR 0.726 vs 0.828 bar) — but the
  hand-trace of the worst doc found the organizer merging 29→9 entities
  ("Dr. Beat" absorbed five albums; "1984" absorbed four years) and exposed
  **GB-11**: entity types were stripped before stage 9 in EVERY mode, so the
  GB-8 type gate had been silently inert since it shipped. Two bugs: coref
  defaulted missing LLM types to UNKNOWN (erasing member types), and the eval
  dump read a nonexistent attribute (`entity_type` vs `.type`) so caches always
  showed `?` — a measurement bug masking a system bug.
- **EXP-TYPE-PROP (GB-11)**: inherit member types in coref; treat UNKNOWN/? as
  blank in gates; fix dumps. ACCEPT. Live: worst doc 9 → 21 correctly-typed
  entities.
- **EXP-SPGOV-2**: better (entR 0.739) but still REVERT — and the trace found
  **GB-12**: distinct years absorbed each other *through the embedding tier of
  semantic dedup* ("1911"/"1939" share ~2/6 trigrams but their embeddings clear
  0.85). The similarity feature added to catch "NYC"/"New York City" was quietly
  merging adjacent years.
- **EXP-LITERAL-GUARD (GB-12)**: distinct digit-content surfaces can never merge —
  a pure string check, zero LLM. ACCEPT: worst doc entR 0.462 → 0.808, 21 → 42
  committed entities.
- **EXP-SPGOV-3**: with all fixes + fan-out: entP 0.808 PASS, relF1 0.161 PASS
  (first time), entR 0.758 best-yet but still short, wall 99 s vs 90 s bar.
  GB-9 stays closed as a default — but every quality metric improved
  monotonically across the three attempts, each driven by a named mechanism.

### Era 5 — speed, milestone, and the frontier of quality
- **EXP-ASYNC-BATCH (GB-4)**: parallel fan-out aimed at verification/coref/
  evidence. −3% (bar −25%) — MISS. The per-call profile showed the *connectivity
  pass* was the dominant serial chunk (~48% of doc wall) — classic Amdahl.
  **EXP-ASYNC-BATCH-2** added it: **−38%** median wall. ACCEPT.
- **EXP-PAPER-COMPARE**: the paper's Table 2 protocol re-run on the local stack.
  Governance delta **+33% strict / +17% mapped** — direction confirmed on a
  different model at $0, magnitude not replicated. Along the way: a real bug
  (ungoverned builds crashed on alias persistence — all 100 flat docs extracted
  then "failed"; fixed + regression-tested), a scorer misuse (mine), and a wedged
  embedding server that motivated the failover.
- **DIAG-SCHEMA-ADMIT**: two offline projections killed two hypotheses in an
  hour: out-of-schema type leak = ceiling ≈ 0 on triple F1 (5.6% of entities,
  barely in scored triples); confidence-floor sweep = strict F1 **flat along the
  entire frontier** (the model's confidences are uncalibrated — thresholding
  trades P for R exactly). Conclusion: the paper-era precision edge was
  extraction/review *quality*, not admission policy.
- **EXP-PAIR-COMPLETE (GB-3)**: offline coverage analysis first — **40% of gold
  pairs are missed while both entities are already in the graph** (ceiling pairR
  0.19 → 0.59). The pair-completion pass hit its pairR bar (0.242 → **0.305**,
  the largest pair-recall gain of any experiment) but REVERTed as configured:
  the model emitted 1,897 triples over ~1,860 candidate pairs — it answered
  every pair despite an explicit "omit unless stated" instruction, diluting
  precision. Next: structural evidence grounding (quote must literally appear in
  the document, enforced in code).

## 5. The distilled lessons

### 5.1 Reverted experiments were the best bug-finders
SP-GOV never shipped, but chasing its failures found two systemic bugs (type
loss, literal merging) that improved *every* mode. Budget for autopsies of
failures, not just celebrations of passes. A REVERT with a hand-trace is worth
more than an ACCEPT without one.

### 5.2 Measurement bugs masquerade as system bugs — instrument both
The eval dump reading a nonexistent attribute made every cached KG show untyped
entities, which simultaneously (a) hid a real type-loss bug and (b) made the
diagnosis look worse than reality. When a field is uniformly degenerate ('?',
0, empty) across all outputs, suspect the *reader* before the writer.

### 5.3 Instruction-following is not a filter
"Omit pairs with no stated relation" → the model linked 1,897 of 1,860
candidates. Any admission decision you care about must be **structural**:
require an evidence quote that literally substring-matches the source, check it
in code, drop non-conforming outputs deterministically. Same principle as the
literal guard: don't ask the model to be careful; make carelessness unrepresentable.

### 5.4 Offline projections before live runs
Cached per-doc artifacts (entities, triples, gold, text, per-call telemetry)
turn week-scale questions into minute-scale ones: the schema-leak ceiling, the
confidence-frontier sweep, and the pair-coverage classification were all
computed for $0 and killed/redirected three goals. Design your harness so every
run leaves behind artifacts an offline script can re-slice.

### 5.5 Confidence scores from LLMs are (by default) not calibrated
Strict F1 was flat along the whole confidence-threshold frontier. Never build an
admission policy on raw self-reported confidence; if you need a precision knob,
build it from evidence checks or a second reviewer, and *verify the frontier
bends* before shipping the knob.

### 5.6 Profile before parallelizing (Amdahl always wins)
The first fan-out attempt parallelized the loops that were easy to see, not the
ones that were expensive (−3%). One per-call wall-time profile later, the fix
was obvious and the same mechanism delivered −38%. Also: the win shrank locally
(−20%) because local calls were already fast — the parallelizable share depends
on the lane.

### 5.7 Dedup/coref must merge, never delete — and gates need real signals
Three separate pathologies (organizer 32→2 collapse; coref dropping singletons;
years absorbing years) shared one shape: a merge step destroying information on
the say-so of a similarity signal. The fixes that held: aliases preserved on
merge, size caps, type gates (fed with *real* types — blank==blank must not
count as a match), and literal guards. Every guard has a fault-injection test
that replays the original pathology.

### 5.8 n=5 tells you "runs/doesn't", never "better/worse"
Run-to-run entity-recall spread at n=5 was ±0.04–0.08 — bigger than most effect
sizes we chased. The hybrid-freeze verdict flipped at n=40. Pre-commit slice
sizes with your bars, and treat small-n wins as hypotheses.

### 5.9 Silent emptiness is the deadliest failure mode
A "successful" run with an empty artifact (valid-but-empty parse, one bad batch,
per-doc exception after extraction) recurred in at least four forms. Defenses
that worked: counting + printing drops at every stage, done-markers, per-item
checkpoints, and treating "0 entities" as an alarm condition rather than a
result. Corollary: exit code 0 means nothing; check the artifact.

### 5.10 Infra fails; runs must not
A wedged embedding server (HTTP up, generation hung) stalled a 100-doc run
indefinitely and, on another night, silently cost a full leg. The sticky
failover (primary ladder exhausted → API embeddings, loud warning) turned the
same wedge into a one-doc delay the very next day. Any external dependency in a
long run needs a tested failover *or* a fast-fail with checkpoint resume.

### 5.11 Fix the constraint, not the symptom — and re-derive priorities after every verdict
The backlog was re-ranked after every experiment: pair discovery stayed the
binding constraint through three eras while "obvious" work (relation naming,
admission thresholds, schema leaks) was killed by cheap diagnostics before
consuming GPU-days. The discipline of asking "what does the evidence say is
binding NOW?" beat any static roadmap.

### 5.12 Local models can carry the architecture
The governance delta survived the move from GPT-5 (~8M tokens/100 docs) to a
local 30B (+33%/+17% vs +47%/+51%, $0). What didn't survive was precision-at-
high-recall — that's review *quality*, the genuine frontier for local-model
governed memory (our GB-14).

## 6. Reference numbers (all from pre-registered runs)

| what | number |
|---|---|
| Production extractor (Qwen3 singlepass) | entR 0.878–0.908, ~8 s/doc |
| Organizer dedup fix (GB-8) | pathology entR 0.244 → 0.801 |
| Type propagation (GB-11) live effect | worst doc 9 → 21 typed entities |
| Literal guard (GB-12) live effect | worst doc entR 0.462 → 0.808 |
| Governed singlepass (SPGOV-3) | entP 0.808 / relF1@0.6 0.161 / entR 0.758 / 99 s |
| Pair completion (GB-3) | pairR 0.242 → 0.305 (ceiling analysis says 0.59) |
| Fan-out (GB-4) | −38% wall API lane, −20% local |
| Freshness (GB-2b) | supersedes ~0 → 1–20/question; first gate pass |
| SciERC governance delta (local) | strict +33%, mapped +17% (paper GPT-5: +47%/+51%) |
| Test suite | 286 (from ~235 pre-loop), incl. fault-injection replays |

## 7. If starting a sibling graphs+memory project tomorrow

1. Build the **offline-scorable artifact cache** and the **experiment log
   discipline** before building features. They compound; features don't.
2. Ship every risky capability **default-off** — your controls become immovable
   by construction and merging to main stops being scary (ours was a pure
   fast-forward after 105 commits).
3. Put **provenance + dates** on facts from day one. Freshness/supersede is a
   prompt-signal problem only if the data survived ingestion.
4. Make merges/dedup **information-preserving** (aliases) from day one, and give
   every guard a fault-injection test replaying a real observed pathology.
5. Assume the model ignores soft instructions at scale; design **structural
   admission** (evidence quotes, schema-constrained decoding, deterministic
   checks in code).
6. Keep a cheap **API experiment lane** routed purely by env vars, and a rule
   that its wins must reproduce locally.
7. Write the verdicts you don't like. The log's REVERT entries are the ones this
   document kept citing.
