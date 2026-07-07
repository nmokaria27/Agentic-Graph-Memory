# LongMemEval — Phase B #1 doctrine

**Goal.** First honest numbers for the FULL stack (extraction + governance + QA/retrieval
layer: PPR, community summaries, AdvancedQAOrchestrator). DocRED measured extraction only;
LongMemEval measures whether the memory system *answers* correctly — including whether it
resolves knowledge conflicts, the system's thesis.

**Paper.** "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
(Wu et al., ICLR 2025). 500 questions, six abilities. 30 questions are abstention
(`_abs` question-id suffix): the correct behavior is refusing to answer.

## Data (gitignored; re-download from HF `xiaowu0162/longmemeval`)

| file | size | contents |
|---|---|---|
| `data/longmemeval_oracle.json` | 15 MB | evidence sessions only per question |
| `data/longmemeval_s.json` | 278 MB | ~115k-token haystack per question (~50 sessions) |

Question fields: `question_id`, `question_type`, `question`, `answer`, `question_date`,
`haystack_dates`, `haystack_sessions` (list of sessions = lists of
`{role, content, has_answer}` turns), `answer_session_ids`.

Type counts (both splits): temporal-reasoning 133, multi-session 133, knowledge-update 78,
single-session-user 70, single-session-assistant 56, single-session-preference 30.

## Harness conventions (mirrors `evaluation/DocRED/`)

- `run_eval.py` — per-question JSON checkpoints (crash loses ≤1 question), cache-skip on
  rerun, instrumented LLM calls, `--max-questions/--offset` into the type-filtered list,
  `--dry-run` for zero-cost slice inspection. One question = one independent context:
  sessions ingested as dated transcripts through `AgentGraphMemoryWrapper`
  (`extraction_mode="wide"` = frozen hybrid v2), then the question is asked once with the
  question date prefixed. Stage-level resume inside a question via `checkpoint_dir`.
- `score_longmemeval.py` — fully offline on cached records. `substring` containment as the
  cheap always-on signal; `--judge` = the benchmark's real metric with the **official
  judge prompts verbatim** (from `xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py`),
  answered by local Nemotron at temp 0, verdict parsed from the last line.
  Substring is None (excluded) for abstention and preference questions.

## Phase ladder (pre-registered)

| phase | slice | gate |
|---|---|---|
| B1-0 smoke | 5 × knowledge-update, oracle | pipeline runs clean end-to-end; hand-read all 5 graphs + answers |
| B1-1 | knowledge-update full (78), oracle | judge acc vs the paper's long-context + commercial baselines; hand-read failures |
| B1-2 | temporal-reasoning 20-slice, oracle | date survival through ingestion (headers → KG) |
| B1-3 | multi-session + single-session slices, oracle | cross-session linking sanity |
| B1-4 | repeat winning slices on `longmemeval_s` | retrieval under real haystack noise (this is the reportable number) |

Rules carried over from DocRED doctrine: never conclude from n=5 beyond "runs/doesn't";
oracle split is diagnostic, `_s` split is the honest benchmark condition; no fix may
reference LongMemEval vocabulary inside `multi_agent_kg/` (bias guard); every slice gets
a written verdict before the next launches.

**Dev/held-out split (pre-registered 2026-07-06, owner's anti-memorization order):**
within each question-type's filtered list, indices **0–19 = development** (may be
hand-read, debugged, iterated on) and **20+ = held-out** (scored at most once per
accepted change, never inspected, never hand-read). All smoke slices draw from dev
indices 0–4. Reportable ability numbers = held-out scores. Any change accepted on a
dev slice must hold on the held-out slice AND not regress one other ability's dev slice
before it counts as a win. The system must improve by adapting to information generally
— never by memorizing this benchmark's questions, formats, or answers.

**Ability order rationale:** knowledge-update first — it directly tests supersede/conflict
resolution in governance, the original reason this system exists. Abstention is scored
within each slice via `_abs` ids rather than as a separate run.

## Cost notes

Oracle knowledge-update questions average ~2 evidence sessions (a few KB) — a full
pipeline build per question is minutes, not hours. `longmemeval_s` is ~460KB of text per
question × 500 questions; B1-4 must be a SLICE (≤20 questions) unless LAZY_INGEST=1 is
validated first.

**Do not launch pipeline runs while the Phase 4 DocRED hybrid leg is on gpu02** — the
extractor code is frozen mid-measurement and GPU contention slows the long pole.
`--dry-run` and the scorer on existing caches are always safe.
