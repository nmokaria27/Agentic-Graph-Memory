# System Hardening Playbook — every scenario, every solution

*Written 2026-07-18 as a deep-thinking pass over the whole system, distilling ~20
experiments of evidence into a scenario → solution map. Companion to
`LESSONS_AND_EXPERIMENTS.md` (what happened) — this is **what could happen and
what to do about it**. Items marked ⚡ are quick wins (<1 day); 🧪 are
pre-registerable experiments; 🏗 are architectural bets. Observed pathologies
cite the experiment that saw them.*

---

## 1. Ingestion & extraction scenarios

**1.1 Silent empty extraction on a valid doc** *(observed: EXP-UNION2, doc_105 —
1 successful call, valid-but-empty parse, no error flag).*
→ 🧪 **Low-yield re-glean trigger** (RESEARCH_IDEAS #2): if a doc's
entities-per-KB or triples-per-entity falls below a corpus-relative floor
(e.g., < 25% of the running median), fire ONE targeted re-extraction and log a
`low_yield_retry` provenance flag. Domain-general: the trigger is statistical,
not content-based. Bar sketch: recovers the doc_105 class on injection tests;
control docs unchanged; ≤1 extra call per triggering doc.

**1.2 List-heavy / enumeration documents** *(observed: doc_109 discography —
over-merge pathology; doc_100 obituary-style enumerations — same-type merges).*
→ The literal guard (GB-12) covers dates/numbers. Remaining exposure: same-type
non-literal enumerations (album titles, sibling names). 🧪 **Merge-evidence
requirement**: an LLM-proposed merge group is honored only if the canonical and
member co-occur as coreferent mentions (apposition/alias pattern) somewhere in
text — checked by substring windows, not by asking the model twice. Expected to
convert most remaining wrong merges into no-ops at zero recall cost (unmerged
duplicates are recall-neutral, precision-cheap).

**1.3 Very long documents — positional recall decay** *(hypothesized by owner;
never measured — EXP-LONGDOC-POSITION is pre-sketched but unrun).*
→ 🧪 Measure recall-by-position curve (whole-doc vs segmented) on synthetic
concatenations of scored docs. If decay is real: overlapping segments with
dedup-by-provenance, or per-segment entity budget forcing attention to tails.

**1.4 Duplicate / near-duplicate ingestion** *(unobserved but inevitable in
production memory use — same page ingested twice).*
→ ⚡ Content-hash idempotency at `process_corpus`: SHA of normalized text →
skip-or-supersede with a provenance note. Without this, every dedup guard gets
stress-tested for no reason and supersede logic sees phantom conflicts.

**1.5 Genre mismatch: conversational text** *(observed: EXP-LMESMOKE era —
verifier rejected 1075/1997 triples as "hallucinated" on chat transcripts; GB-6
parked).*
→ 🧪 Hand-read 30 rejected triples from smoke caches (cheap, offline). If the
verifier is genre-biased, add genre-neutral framing: verification should ask
"is this supported by the conversation" not "is this stated in the document."
One prompt-level change, controls on DocRED must not move.

**1.6 Tables/structured content inside text** *(unobserved; likely in real
corpora).*
→ Defer until seen; the DocProcessor segment layer is the insertion point.
Record here so the future session recognizes the symptom: entity recall
collapse specifically on docs containing markup/tabular runs.

## 2. Model scenarios

**2.1 Model swap changes everything silently** *(observed 3×: Nemotron→Qwen3
changed dedup aggressiveness; kimi returned nested fragments = silent 0-entity
docs; gemma4 was loaded with zero harness validation).*
→ ⚡ **Preflight assert in every run script**: query `/v1/models`, assert the
served id equals the expected id, assert embedding endpoint returns the expected
dimension, echo both into the log header. (Tonight's gemma4 surprise would have
been caught in 1 second.)
→ 🧪 **Model qualification gate** (formalize EXP-MODEL-LOCAL as a template):
slice-A extraction + 5-question LongMemEval smoke + JSON-shape suite through
the project client. No model becomes production without passing; results append
to the model-quirks catalog in SERVER_GUIDE.

**2.2 Instruction non-compliance at scale** *(observed decisively:
EXP-PAIR-COMPLETE — 1,897 triples over ~1,860 pairs despite an explicit omit
instruction).*
→ Principle, now proven twice (literal guard, evidence grounding): **make the
constraint structural, never instructional.** Catalog of structural gates to
reach for: verbatim-quote substring checks (GB-3b), enum-constrained decoding
(GB-5), degenerate-endpoint filters, size caps, literal guards. Any new
admission decision should cite which structural gate enforces it.

**2.3 Uncalibrated confidence** *(observed: DIAG-SCHEMA-ADMIT — strict F1 flat
along the entire confidence frontier).*
→ Never build policy on raw self-reported confidence. If a precision knob is
needed: 🏗 replace confidence with **evidence multiplicity** (count of
independent verbatim quotes supporting a triple — monotone, checkable, and
meaningful across models). 🧪 cheap validation: recompute the DIAG frontier
using quote-count instead of confidence on a grounded cache; if the frontier
bends, the knob is real.

**2.4 Thinking-model CoT leakage into JSON** *(observed: Nemotron era — raw
reasoning prose where JSON was expected).*
→ Already guarded by Pydantic-validate-or-degrade at every boundary (GB-1).
Keep `LLM_THINKING_MODELS` env override on the GB-5 list so new thinking models
don't need code edits.

## 3. Graph integrity scenarios

**3.1 Destructive merges are irreversible** *(structural weakness underlying
GB-8/11/12 — every fix so far prevents bad merges; none can undo one).*
→ 🏗 **SAME_AS edges instead of physical merges**: keep both entities, add a
governed identity edge; queries resolve identity clusters at read time. Merges
become reversible governance decisions with provenance, and a wrong merge is a
one-edge revert instead of graph surgery. Cost: read-path union-find (cheap,
cacheable). This is the single highest-leverage architectural change for a
memory system that must be trusted long-term.

**3.2 Value-type collisions — the q0 pattern** *(observed 3×: `HAS_TIME` /
`HAS_VALUE` both active for the same fact; relation names not semantically
close so GB-2c can't catch it).*
→ 🧪 **GB-2d design (write before code, per the standing slip lesson)**:
classify triple objects into shape classes (duration, date, money, count,
percentage, name) with regex/parsers — no vocabulary. Per subject, if ≥2 ACTIVE
triples hold objects of the SAME shape class via different relations, emit a
conflict candidate to the resolver with both shown. Bar: q0-class collisions
surface as conflicts on the cached LongMemEval KGs (offline replay possible!)
without new conflicts on DocRED controls.

**3.3 Open-world relation-name proliferation** *(latent: every corpus mints
relations; `worked_for` vs `WORKED_FOR` both admitted — seen in paircomp logs).*
→ ⚡ Case/separator normalization at propose-time (pure string, zero risk).
→ 🧪 Periodic **relation canonicalization pass** mirroring entity aliases:
embedding-cluster relation names per domain, record `relation_aliases` table,
apply at query time only (storage keeps originals — reversible, auditable).

**3.4 Dangling refs / structural invariants after partial failures** *(observed
pre-GB-8: 27/29 endpoints dangling after mass deletion).*
→ ⚡ **`graph fsck`**: standalone script asserting every triple endpoint
exists, every alias target exists, no superseded triple is also active, every
active triple has ≥1 provenance ref. Run automatically at the end of every
benchmark; print violations; never auto-fix silently. Turns a class of silent
corruption into a loud postcondition.

**3.5 Unbounded growth / stale mass** *(unobserved at current scale; certain at
memory-system scale).*
→ 🏗 Compaction policy: superseded triples exported to cold storage after N
generations; per-domain size budgets trigger a summarize-and-archive pass
(community summaries already exist as the mechanism). Design only when a real
corpus hits the wall — but record the trigger metric now: QA latency or
resolver context overflow.

## 4. Governance & review scenarios

**4.1 Local-model under-pruning — the paper-precision gap** *(observed:
EXP-PAPER-COMPARE keeps 68% of flat triples vs paper's 37%; DIAG killed
thresholding).* Three candidate mechanisms, in cost order:
1. 🧪 **Grounded verification everywhere** — extend GB-3b's verbatim-quote gate
   from pair-completion to the main verification pass (verifier already returns
   evidence; today nothing checks it). Prediction: rejects the same triples a
   GPT-5 reviewer rejects, at $0. This is the highest-expected-value experiment
   in the queue.
2. 🧪 Two-tier review: borderline triples (no quote, partial verification) get
   ONE adjudication call at LARGE tier; clear passes/fails don't.
3. 🧪 Model headroom (EXP-HEADROOM-SCIERC running now decides if this matters).

**4.2 Governance modes never exercised at scale** *(permissive used everywhere;
strict/triage validated only on small governance benchmarks).*
→ 🧪 After GB-14: one SciERC build in `triage` mode vs permissive — does
routing/escalation change admitted-graph quality, or only audit richness?
The paper claims governance value; the mode dimension is the untested half.

**4.3 Domain skew — one mega-domain absorbs the corpus** *(partially observed:
doc-level domain configs give 3–5 domains; corpus-level skew unmeasured).*
→ ⚡ Log per-domain entity counts + Gini at end of corpus builds (pure
reporting). 🧪 If skew is real: re-bootstrap trigger when the top domain owns
>70% of entities.

## 5. Memory & QA scenarios

**5.1 Abstention — answering when the graph doesn't know** *(GB-7, parked; the
`_abs` LongMemEval slices exist for exactly this).*
→ 🧪 Retrieval-sufficiency gate: answer only if the assembled evidence subgraph
is non-empty AND covers the question's entities; otherwise emit "not in
memory." Measure: abstention-slice accuracy up, non-abstention slices unmoved.

**5.2 Conflicting sources, no date signal** *(freshness handles dated conflicts;
undated contradictions currently coexist silently).*
→ QA-side: when the evidence subgraph contains an unresolved conflict, the
answer should surface it ("sources disagree: X (doc A), Y (doc B)") rather than
picking one. ⚡ in the answer assembler; huge trust win for a memory product.

**5.3 Temporal reasoning beyond freshness** *(LongMemEval ability untested —
"what did I do BEFORE X?").*
→ Depends on document_date already being in provenance (done). 🧪 The breadth
run (tonight) characterizes it; expected gap: QA context assembly doesn't
expose dates to the answerer. Fix is assembler-side (attach `as of <date>` to
each fact line), not storage-side.

**5.4 Cross-session identity** *("my sister" in session 3 = "Anna" in session 40).*
→ Cross-document alias sync exists and is now crash-proof (incident 2 fix) but
its QUALITY is unmeasured. 🧪 LongMemEval multi-session ability is the natural
probe; if weak, the SAME_AS architecture (3.1) is the principled home.

**5.5 Query-time cost** *(QA latency unmeasured since GB-10 fix).*
→ ⚡ Add wall-time to requery harness output. Defer optimization until a number
exists (the GB-4 lesson: profile before parallelizing).

## 6. Infrastructure scenarios

**6.1 LLM server wedge mid-run** *(observed for embeddings — gpu01; the vLLM
analog WILL eventually happen).*
→ ⚡ Watchdog: cron script curling `/health` every 5 min, auto-restart with the
SERVER_GUIDE recipe + loud log line. The embedding failover pattern
(sticky, loud, fail-safe) is the template if a second LLM lane ever exists.

**6.2 Server up, garbage out** *(near-miss: wrong model loaded; also FP8 +
sampler quirks can silently degrade).*
→ ⚡ **Canary check at run start**: one fixed known-answer extraction call; if
the parse fails or entities=0, abort the run BEFORE burning hours. Pairs with
the preflight model-name assert (2.1).

**6.3 Concurrent runs clobbering caches / competing for GPU** *(observed: 5
stray pytest suites; two suites contending → 4h "hang").*
→ ⚡ Lockfile per cache dir (`flock` in run scripts, refuse to start if held).
⚡ pkill hygiene: always `pkill -f "[p]attern"` (self-match killed our own
shells 3× — exit 144).

**6.4 Disk growth on /scratch and results/** *(checkpoints × runs × models).*
→ ⚡ A `make gc` target: delete cache dirs older than N days whose verdicts are
written (the log is the source of truth, not the caches — R5 already says so).

**6.5 Session/agent death mid-run** *(designed-for and validated: nohup +
done-markers + per-item checkpoints + STATUS: RUNNING entries in the log).*
→ Keep the fresh-agent checklist: FIRST scan EXPERIMENT_LOG for `STATUS:
RUNNING` without a verdict and score those before anything else.

## 7. Measurement scenarios

**7.1 Wrong scorer / harness misuse** *(observed: evaluate_kg.py vs
score_accumulated_scierc_rich.py — "1/100 docs" nonsense).*
→ ⚡ One documented scoring entrypoint per benchmark in BENCHMARKS_GUIDE; each
scorer asserts the artifact shape it expects and refuses otherwise.

**7.2 Lossy dumps hide system state** *(observed twice: `.entity_type` attr bug;
source/evidence stripped from triples).*
→ ⚡ **Round-trip test**: build a KG with every field populated → dump → assert
every Entity/Triple field survives. One test kills the whole class.

**7.3 Noise floor unknown per metric** *(known for entR at n=5: ±0.04–0.08;
unknown for pairR/relF1 at n=20).*
→ 🧪 One A/A run (same config, two seeds/draws, n=20) to publish the noise
floor table; every future bar should clear the floor, not just the baseline.

**7.4 Held-out hygiene over long projects** *(slice B untouched; LongMemEval
20+ indices reserved — but only discipline enforces it).*
→ ⚡ A `HELD_OUT.md` manifest listing every reserved slice + the rule; CI grep
that no run script references held-out indices except the sanctioned one.

## 8. The prioritized queue (synthesis)

Quick wins, do in one sitting (~1 day total): preflight+canary (2.1/6.2),
run-lock + gc (6.3/6.4), fsck (3.4), dump round-trip test (7.2), relation-name
normalization (3.3⚡), ingestion idempotency (1.4), conflict surfacing in
answers (5.2), HELD_OUT manifest (7.4).

Experiment queue (each pre-registered, one mechanism):
1. **Grounded verification everywhere** (4.1.1) — likely the biggest quality
   jump left; directly attacks the paper-precision gap with the mechanism GB-3b
   just validated.
2. **GB-2d value-shape collisions** (3.2) — closes the last confirmed thesis-gap
   pattern; offline-replayable on cached KGs.
3. **GB-5 structured decoding** — retires three failure classes at once (JSON
   drift, schema leaks, type enums).
4. **Low-yield re-glean** (1.1) + **merge-evidence requirement** (1.2) — recall
   and precision siblings for extraction.
5. **LongMemEval breadth** (running/tonight) → its failure modes re-rank
   everything below.
6. GB-6 genre verification, GB-7 abstention, A/A noise floor, triage-mode
   benchmark.

Architectural bets (design docs before code, in value order):
1. **SAME_AS reversible identity** (3.1) — the trust foundation for long-lived
   memory.
2. **Evidence-multiplicity replacing confidence** (2.3) — a calibratable signal
   the whole pipeline can share.
3. **Quarantine commit** — per-document staging graph promoted atomically after
   governance; makes bad-document rollback one operation (pairs with 3.1).
4. **Query-driven densification** — run expensive passes (pair-completion,
   re-glean) only on subgraphs QA actually touches; converts recall spend from
   corpus-uniform to demand-weighted.
5. Compaction/archival (3.5) when scale demands.

## 9. The one-paragraph doctrine for whoever picks this up

Pre-register before running; one named mechanism per change; controls
default-off and immovable; offline projection before live GPU; structural gates
over instructions; verdicts honest including reverts; profile before
optimizing; check the artifact, never the exit code; and when a run fails,
autopsy it — this codebase's best fixes were found inside its failed
experiments.
