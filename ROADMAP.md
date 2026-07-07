# ROADMAP & Session Handoff — Agentic Graph Memory

**Purpose of this file:** complete onboarding for any coding agent (or human) picking this
project up. It records the goals, everything done in the July 2026 experiment sessions, every
fix with its commit, what is running right now, hard constraints, and the exact next steps.
Read this first; follow the doc index at the bottom for depth.

---

## 1. Project goals (the owner's standing requirements)

1. **Robust system** — benchmark runs must never fail silently mid-run again (the project's
   origin story: a MemoryAgentBench run collapsed from 122 extracted entities to 9 in the KG).
2. **Domain-adaptive** — NO fixed general-purpose extractors (GLiNER explicitly rejected).
   `DomainClassifier` discovers the schema per corpus; the system must self-organize for law /
   medical / any incoming domain.
3. **Quality first** — "rich, solid and good" KG data beats speed. Longer runs are acceptable.
4. **No benchmark bias** — fixes must be structural/domain-general, never tuned to a dataset's
   vocabulary. Guards: held-out slices, untouched controls, cross-benchmark validation.
5. **Experiment → document → decide** — every change gets defined before running, measured
   against pre-registered bars, and written up honestly (misses reported as misses).
6. Local models only: **Nemotron-30B FP8 on vLLM (gpu02)**, embeddings `mxbai-embed-large`
   on **Ollama (gpu01)**. See `SERVER_GUIDE.md` for cluster rules.

## 2. System in one paragraph

Multi-agent governed KG pipeline (9 stages): DocumentProcessor → DomainClassifier (adaptive
schema) → EntityExtractor (multi-stage or **wide-harvest** mode) → RelationExtractor (RHF:
relation-head-first multi-stage; now also consumes wide-harvest seed candidates) →
Connectivity → EvidenceLinking → Deliberation → Verification → KnowledgeOrganizer (stage 9:
dedupe, coref-aware integration, governance commit into `GovernedKnowledgeGraph`). QA layer
(`AdvancedQAOrchestrator`, PPR retrieval, community summaries) sits on top — **not yet
benchmarked** (that's Phase B). Key dirs: `multi_agent_kg/` (system), `evaluation/` (harnesses),
`tests/` (235 passing as of commit `c16b55d`).

## 3. What happened in these sessions (chronological, with commits)

### 3.1 Root-caused the original failure ("122 → 9")
Not a filter bug. Thinking models (Nemotron) can burn the whole completion budget on hidden
reasoning and return EMPTY content with `finish_reason=length`; `chat_completion` silently
returned `""` and stages swallowed it. Fixes:
- **Truncation-retry ladder** (`multi_agent_kg/llm/openai_client.py`, commit `738cf3b`):
  detect empty+length, double budget, retry.
- **Prompt-aware cap + 65536 ceiling** (commit `419d049`): ladder previously requested
  budget+prompt > context window (vLLM 400) and one 92-min ladder climb; cap now subtracts
  estimated prompt tokens (`chars/3`), env `NEMOTRON_BUDGET_CEILING` (default 65536).
- `NEMOTRON_THINKING=on` is REQUIRED — reasoning-off yields blank extraction on this build.

### 3.2 Extraction strategy experiments (`EXTRACTION_EXPERIMENTS.md`)
- RHF (multi-stage) vs GraphRAG-style singlepass on a 12-fact doc → graph dumps showed RHF
  richer/more-correct per fact, singlepass cheaper with better connectivity. Led to the
  **hybrid** idea: singlepass-style wide harvest front end + RHF deliberation back end.
- **Self-consistency (SC) rewritten** (commit `738cf3b`): stock SC voted on exact whole-response
  strings (always disagree at temp>0 → arbitrary sample at 1/n confidence). Now item-level
  consensus (union across samples, identity-keyed dedupe excluding `type` and sample-local
  `id`, support-blended confidence). **SC remains GATED** — live run failed on temp-0.7
  reasoning runaway; re-validation is a pending task. Do NOT enable in benchmark configs.
- **Coref collapse** (doc could shrink 29 entities → 1): coref may merge but NEVER delete
  (commit `60ca03e`). Reproduced without SC; structural fix; no collapse in 45+ runs since.

### 3.3 Re-DocRED evaluation harness (`evaluation/DocRED/`, commit `5219908`)
- **Why Re-DocRED**: original DocRED misses ~60% of true triples → punishes open-world
  extractors. Data: `data/dev_revised.json` (500 docs), `train_revised.json`,
  `rel_info.json` (95 P-code→name map built from the Wikidata API — GitHub mirrors 404).
- `run_eval.py`: per-doc JSON checkpoints (crash loses ≤1 doc), cache-skip on rerun,
  instruments every LLM call, dumps predicted graph + gold + entity `labels`.
- `score_docred.py`: layered offline scorer — L1 entity (name ∪ labels vs gold mention
  clusters), L2 pair recall (direction flips counted separately), L3 relation match
  (embedding sim @0.6/0.7/0.8; 0.6 is the honest threshold for open schemas).
- Doctrine (`PLAN.md`): slice A = docs 0–4 (diagnostic, hand-read), slice B = docs 30–34
  (held-out, never tuned on), singlepass = untouched control. Fix → re-run same slice →
  accept/revert. Lessons in `LESSONS.md`.

### 3.4 The matrix arc (all verdicts in `MATRIX_REPORT.md`, newest first)
| matrix | change tested | headline |
|---|---|---|
| v1/v2 | coref fix, value-entity policy (`c7f36b3`) | coref fix = generalizing win; hybrid v1 leaked 15–30% entities at integration; `wide_relation_candidates` harvested but unconsumed |
| v3 | **stage-9 funnel fixes + relation seeding** (`5c81a79`) | seeding decisive: hybrid pairR +76%, relF1 2×; entity-recall bar MISSED (honest) |
| v4 | **labels measurement fix** (`2a7446a`) | ~2/3 of the "recall gap" was a scoring artifact (coref canonical ids vs gold surfaces in `labels`); hybrid entR 0.79 ≈ singlepass 0.81 on slice A |
| v5 | **keep unreferenced DATE nodes** (`d47e9bc`) | held-out pairR 0.186→**0.245** (+32%, beats singlepass); **n=5 noise floor reached** (rhf swung ±0.08 with unchanged code) |

**Standing decision (v5): hybrid v2 is the frozen production extractor** — best relation
quality (relF1 0.211) + pair recall + governance; singlepass stays as cheap bulk-recall mode.
`extraction_mode="wide"` on `DeliberativeOrchestrator` = hybrid; strategy flags in `run_eval.py`.

### 3.5 Stage-9 fixes detail (commit `5c81a79`, `knowledge_organizer.py`)
- id-cleanup regex `_\d+$` ate 4-digit YEAR suffixes (collapsed distinct events) → now `_\d{1,2}$`.
- same-id collisions silently DISCARDED the second entity's aliases → now merged via
  `add_entity` (which merges labels for existing ids).
- every drop has a counted reason (`entity_drop_reasons`), printed + in `integration_stats`.
- unreferenced pure integers: keep if `classify_value`→DATE (a year is a queryable memory node);
  drop bare NUMBER counts (`d47e9bc`).

### 3.6 Bias audit (owner asked explicitly)
Zero DocRED contamination in `multi_agent_kg/` (no P-codes, no gold schema; verified by grep +
git blame). All system changes are instrumentation, bias *removals*, or reuse of the system's
own output. Scorer changes are measurement-side and control-verified (singlepass identical
across v3/v4/v5). **Pre-existing bias NOT from these sessions**: `_GARBAGE_PHRASES` and
`_TYPE_CONSOLIDATION` in `knowledge_organizer.py` are medical/scientific-leaning lists from
commit `3bc4b93` — flagged for a future domain-neutrality pass. Known imperfection: id regex
still strips legit 1–2 digit suffixes ("Apollo 11").

## 4. RUNNING RIGHT NOW (as of 2026-07-06)

**Phase 4 confidence run** — PID 949154, script `evaluation/DocRED/phase4_confidence.sh`,
log `evaluation/results/phase4_confidence.log`.
- **Code version: commit `c16b55d`** (= frozen hybrid v2: ladder fixes + funnel fixes +
  seeding + labels dump + DATE keep). Strategies: singlepass first (control, ~2 h), then
  hybrid (~12 h), on **40 fresh never-touched docs (offset 100–139)**, cache dir
  `evaluation/results/docred_kg_cache_phase4/`.
- Outputs: `docred_phase4_<strategy>.json` (run summaries),
  `docred_scores_phase4_<strategy>.json` (offline scores).
- Monitor armed on `PHASE4_DONE`. Zero-error expectation; per-doc checkpoints mean a crash
  loses ≤1 doc; reruns skip cached docs.
- **Purpose**: n=40 kills the n=5 variance problem; confirms hybrid≈singlepass entR + better
  pairR/relF1 at scale; freezes the extraction baseline for Phase B.

### HARD CONSTRAINT while Phase 4 runs
**Do NOT edit `multi_agent_kg/`** until the hybrid milestone appears in the log. The script
launches the hybrid process AFTER singlepass finishes; edits now would be picked up by that
fresh process and invalidate the "frozen extractor" measurement. Safe to touch: `evaluation/`
(new adapters/scorers), `tests/` (new tests may run against edited eval code only), docs.

## 5. Safe parallel work (ordered; all eval-side)

1. **LongMemEval prep (Phase B #1)**: download dataset → `evaluation/LongMemEval/data/`;
   build `run_eval.py` mirroring DocRED conventions (per-context checkpoints, offline scorer,
   `--max-samples/--offset`, cache-skip). Ability splits: knowledge-updates FIRST (tests
   supersede/conflict-resolution — the system's thesis), then temporal, multi-session,
   single-session, abstention. 5-question smoke slices before anything bigger.
2. **MAB adapter resume** (Phase B #2 prep): wire `checkpoint_dir` through
   `evaluation/Memory-Agent-Bench/agent_graph_memory_adapter.py` (`_run_pipeline`) so
   per-context resume works; the orchestrator already supports checkpoints (`ckpt.save/load`).
3. **Direction-flip analysis (offline)**: mine v4/v5 caches for flipped pairs; classify which
   relations invert (BORN_IN, LOCATED_IN…); design the post-check; implement only after
   Phase 4 finishes.
4. **LLM-judge relation scorer (offline)**: add `--judge` mode to `score_docred.py` scoring
   predicted-vs-gold relation *meaning* with Nemotron on existing caches — closes the
   embedding-threshold measurement gap. Runs on cached predictions; no pipeline involvement.
5. **SC re-validation prep**: plan = temp ~0.4, explicit max_tokens in
   `call_llm_with_self_consistency`; execute AFTER Phase 4 (touches system code).

## 6. Next steps after Phase 4 (the full ladder)

- **Phase 4 verdict** → append to `MATRIX_REPORT.md`; if slice-A picture holds at n=40,
  DocRED closes (remaining items #3/#4 above ride on its data).
- **Phase B — memory benchmarks (the original goal)**: LongMemEval smoke → ability slices →
  verdicts; then **re-run the exact MemoryAgentBench config that originally collapsed** as the
  regression proof. First honest numbers for the QA/retrieval layer (PPR, community summaries).
- **Phase C — positioning**: MuSiQue + 2WikiMultiHopQA subsets (100–200 Qs) vs published
  HippoRAG/LightRAG numbers; UltraDomain (legal/agriculture/CS) for the domain-adaptivity
  claim AND as the DocRED-overfit guard. STaRK-Prime optional (retrieval-only).
- **Phase D — self-improvement loop**: Tier 1 = automate fix→re-measure over prompt/config
  variants using the DocRED harness as reward; Tier 2 = benchmark lessons stored as retrievable
  exemplars consulted per domain; Tier 3 (research) = DPO-LoRA on Nemotron from
  deliberation/governance accept/reject pairs collected during Phase B/C.

## 7. Gotchas & environment notes (hard-won)

- `NEMOTRON_THINKING=on` always; reasoning-off ⇒ empty extraction. Budget env:
  `NEMOTRON_BUDGET_CEILING` (65536), `VLLM_MAX_MODEL_LEN=131072`.
- **Pathological docs exist** (e.g. dev doc 33 "Kyoto Imperial Palace", doc 1 "Ross Alger"):
  reasoning runaway → ladder rescues but wide-call yield drops. Candidate future fix:
  low-yield re-glean. Segment errors are contained; never fatal.
- rhf strategy is variance-heavy run-to-run (±0.08 entR at n=5) — never conclude from small
  rhf deltas.
- `evaluation/results/` is **gitignored** — verdicts must be written into reports.
- `gh` lives at `~/miniconda3/bin/gh` (not on default PATH); fine-grained PAT needs Contents:
  Read-and-write. Pushes: `git push origin feat/vector-index-and-qa-improvements`.
- After ANY `multi_agent_kg/` change: `graphify update .` (project rule) + full
  `python -m pytest -q` (235 passing baseline).
- Run scripts: always `nohup bash <script> > /dev/null 2>&1 &` + `until grep -q <MARKER>` wait
  loops; per-doc checkpoints + cache-skip make reruns safe. Delete stale strategy caches when
  the pipeline changes (scripts do this).
- mem0 plugin SDK is broken on this host; durable memory = repo docs + `~/.claude/.../memory/`.
- **Fireworks lane (added 2026-07-06):** `FIREWORKS_API_KEY` in `.env`; any run can target
  Fireworks per-process via env overrides (`LLM_BACKEND=vllm`,
  `VLLM_BASE_URL=https://api.fireworks.ai/inference/v1`, `VLLM_API_KEY=$FIREWORKS_API_KEY`,
  `LLM_DEFAULT_MODEL=accounts/fireworks/models/<m>`). Experiment lane ONLY — production
  stays local (requirement #6); see `.claude/skills/self-improve/SKILL.md` §5.

## 8. Document index

| doc | contents |
|---|---|
| `EXPERIMENT_LOG.md` | self-improvement loop: every experiment pre-registered + verdict (newest at bottom) |
| `.claude/skills/self-improve/SKILL.md` | the loop agent's guide: context bootstrap, doctrine, Fireworks lane, discipline |
| `RESEARCH_IDEAS.md` | 25 vetted techniques w/ experiment sketches (deep-research pass 2026-07-06) |
| `evaluation/DocRED/MATRIX_REPORT.md` | all matrix verdicts v1→v5, newest first (single source of truth) |
| `evaluation/DocRED/LESSONS.md` | per-failure-mode lessons from hand-reading graphs vs gold |
| `evaluation/DocRED/PLAN.md` | the phased doctrine (slices, gates, scoring design) |
| `evaluation/DocRED/HYBRID_V2_RUN.md` | pre-registered hybrid v2 changes + success bars |
| `EXTRACTION_EXPERIMENTS.md` | strategy experiments, SC status + gates, profiling history |
| `SERVER_GUIDE.md` | cluster rules (vLLM gpu02 / Ollama gpu01, ports, scratch) |
| `evaluation/BENCHMARKS_GUIDE.md` | LoCoMo + MemoryAgentBench how-to |
| `AGENT_SCHEMA.md`, `ARCHITECTURE_COMPARISON.md` | system design references |
