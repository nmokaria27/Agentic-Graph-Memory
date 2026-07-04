# Roadmap — from matrix v3 to a self-improving, benchmark-proven memory system

Anchored to the original goals: a **robust** multi-agent KG memory system (no mid-run
failures), **domain-adaptive** (self-organizing schema, no fixed extractors), **quality-first**
("rich, solid and good" KG data over speed), proven on benchmarks that previously crashed —
and, longer-term, a system that improves itself from its own evaluation signals.

Status marker: matrix v3 running (hybrid v2 fixes). Everything below assumes its verdict.

## Phase A — close out DocRED (extraction quality) · ~2-3 days

1. **Matrix v3 verdict** (tomorrow): if hybrid v2 meets its bars (HYBRID_V2_RUN.md — entR
   ≥0.75/0.80, pairR ≥ singlepass, flips ≤2), **hybrid becomes the production extractor**
   with singlepass/rhf behind flags. If not: one more fix loop, then decide.
2. **Deferred fixes, one loop each** (fix → re-run same 5 docs → accept/revert):
   - direction post-check (flip detector on asymmetric relations like BORN_IN / LOCATED_IN)
   - remaining TIME/NUM recall gap (gold's date/quantity nodes; policy already landed)
   - LLM-judge relation scoring pass (open-schema names get punished by embedding
     thresholds — a judge model closes the measurement gap, not the system gap)
3. **Phase 4 confidence run**: 30–50 held-out dev docs, best config, overnight. Gate: slice-A
   trends hold. This freezes the extractor for the memory benchmarks.
4. **SC re-validation** (gates in EXTRACTION_EXPERIMENTS.md): coref never-delete landed and
   `id`-identity fixed — re-run `rhf --sc` on the 12-fact doc with SC temp ~0.4 + explicit
   max_tokens. If it passes, SC becomes a quality dial for governance-critical ingestion.

## Phase B — memory benchmarks (the original goal) · ~1 week

5. **LongMemEval smoke → slices** (the best diagnostic for THIS system):
   download, adapter (mirror DocRED runner conventions: per-context checkpoints, offline
   scoring), then 5-question slices per ability. Run order: knowledge-updates first (tests
   supersede/conflict-resolution — the system's thesis), then temporal, multi-session,
   single-session, abstention. Each ability slice = its own lessons file before any full run.
6. **Return to MemoryAgentBench** — the benchmark that originally collapsed (122→9).
   The root causes are all fixed (truncation ladder, partial-KG preservation, coref).
   Remaining wiring: pass `checkpoint_dir` through the MAB adapter for per-context resume.
   Re-run the exact configuration that failed, as the regression proof of "robust system."
7. **QA-layer eval**: DocRED scores extraction only. LoCoMo/LongMemEval exercise
   retrieval + QA orchestration end-to-end — PPR retrieval, community summaries, and the
   answer formatter all get their first honest numbers here.

## Phase C — comparison & positioning · ~1 week, parallelizable

8. **MuSiQue + 2WikiMultiHopQA subsets** (100–200 questions): build-KG-from-corpus +
   multi-hop QA — the eval where HippoRAG/LightRAG publish numbers. This is the public
   "is my system competitive" answer.
9. **UltraDomain (legal/agriculture/CS)**: directly exercises the domain-adaptivity claim —
   DomainClassifier must reshape the schema per corpus with zero config. Also the guard
   against DocRED overfitting.
10. **STaRK-Prime** (optional, retrieval-only): stress the retrieval layer on a given KG.
    Only if Phases A-B leave appetite; it bypasses extraction/governance entirely.

## Phase D — self-improvement loop (the RL question, made practical)

11. **Tier 1 — automated config/prompt search**: the DocRED harness is now a cheap, repeatable
    reward signal. Wrap it: candidate change (prompt variant, threshold, stage toggle) →
    5-doc slice → accept if F1 ↑. This is the manual Phase-3 loop, automated. No training.
12. **Tier 2 — lessons as memory**: store benchmark failure patterns as retrievable exemplars
    the extraction prompts consult per domain (the system eating its own dogfood). Zero
    training cost, fully aligned with the self-organizing thesis.
13. **Tier 3 — DPO-LoRA from governance signals** (research track): deliberation/verification
    votes already label accepted vs rejected extractions; collect pairs during Phase B/C runs,
    then DPO a LoRA on Nemotron. Only after Tiers 1-2 plateau — and only with the eval
    harnesses as the referee.

## Standing infrastructure items (do opportunistically)

- **GitHub push auth** (blocked: no gh/SSH on host) — install gh + `gh auth login`, then push
  the backlog of local commits.
- **DomainClassifier cost**: item-level consensus should now produce real confidence (less
  perpetual escalation); measure, and cache the domain per corpus (~90 s/build back).
- **Graph/report hygiene**: keep `graphify update .` post-change; MATRIX_REPORT + LESSONS
  stay the single source of truth for verdicts.

## Sequencing logic

DocRED freezes the extractor → LongMemEval/MAB prove the memory system end-to-end → MuSiQue/
UltraDomain position it against the field → the self-improvement loop turns all those harnesses
into training signal. Each phase's benchmark doubles as the regression suite for the next.
