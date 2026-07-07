# EXPERIMENT LOG — self-improvement loop

**Rules (from `.claude/skills/self-improve/SKILL.md`):** every experiment and every code
change gets an entry — pre-registration BEFORE running, verdict appended AFTER scoring,
failures included. Newest entries at the BOTTOM. Fireworks-lane results are evidence,
not reportable numbers; local-lane DocRED verdicts also go to
`evaluation/DocRED/MATRIX_REPORT.md`. `evaluation/results/` is gitignored — numbers only
survive if written here.

Test baseline: **235 passing** (`python -m pytest -q`).

---

## EXP-1: Model-headroom baseline — slice A on Fireworks kimi-k2p6  (2026-07-06)
- **Hypothesis:** If a stronger open-weight model substantially lifts slice-A extraction
  (esp. hybrid entR and the zero-triple funnel behavior), the current ceiling is
  model-bound, not pipeline-bound — which reprioritizes G3/G5 (pipeline fixes) vs
  "wait for better local model". Also validates the entire Fireworks lane end-to-end.
- **Change:** none (pure env-override model swap; code at commit 472ad4e).
- **Lane & model:** fireworks `accounts/fireworks/models/kimi-k2p6`.
- **Slice & control:** docs 0–4 (slice A), strategies singlepass + hybrid. Control =
  Nemotron slice-A numbers from matrix v4/v5 (MATRIX_REPORT.md): hybrid entR 0.79 /
  singlepass 0.81; hybrid relF1@0.6 ≈ 0.21.
- **Success bar (interpretive, n=5):** kimi hybrid entR ≥ 0.85 AND relF1@0.6 ≥ 0.25 ⇒
  "model-bound" signal; within ±0.05 of Nemotron ⇒ "pipeline-bound" signal. Any
  zero-triple doc on kimi hybrid ⇒ strong evidence the funnel bug is pipeline logic (G3).
- **Cost estimate:** ~150–250 Fireworks calls, ~30–60 min wall.
- STATUS: RUNNING — `evaluation/DocRED/fw1_kimi_sliceA.sh`, log
  `evaluation/results/fw1_kimi_sliceA.log`, caches
  `evaluation/results/docred_kg_cache_fw_kimi/`, marker `FW1_DONE`.

### EXP-1 amendment: kimi-k2p6 aborted, retargeted to glm-5p2  (2026-07-06)
- **Finding (unplanned, valuable):** on kimi-k2p6, `chat_completion_json` returned a
  nested fragment (`{source, relation, target}`) instead of the full document → doc 0
  singlepass silently produced 0 entities. Plain `chat_completion` on the same prompt
  returned perfect JSON — the failure is in the client's per-model JSON-mode/extraction
  heuristics (`_THINKING_MODEL_PATTERNS` doesn't know kimi; no env override exists).
  kimi-k2p5 500s server-side. glm-5p2 and gpt-oss-120b work correctly through
  `chat_completion_json` (5-6 ents on the probe).
- **Implication:** direct evidence for G4 (RESEARCH_IDEAS #1 guided JSON): per-model
  string heuristics are brittle; schema-enforced decoding removes the class. Client is
  frozen (Phase 4) — after the freeze, add kimi patterns AND/OR guided JSON, plus an env
  override (`LLM_THINKING_MODELS`) so model onboarding never needs a code edit.
- **Action:** killed kimi processes (Phase 4 PID verified alive), deleted contaminated
  cache, script renamed `fw1_glm_sliceA.sh`, model → `accounts/fireworks/models/glm-5p2`,
  cache → `docred_kg_cache_fw_glm/`. Success bars unchanged.
- STATUS: RUNNING — log `evaluation/results/fw1_glm_sliceA.log`, marker `FW1_DONE`.

---

## EXP-2: LongMemEval B1-0 smoke via Fireworks (goal G1)  (2026-07-06)
- **Hypothesis:** the LongMemEval harness (runner → AgentGraphMemoryWrapper ingest →
  AdvancedQA answer → offline scorer) works end-to-end; hand-reading 5 knowledge-update
  answers reveals where the QA layer loses updated facts (thesis ability, B1-0 gate:
  "runs clean end-to-end" per evaluation/LongMemEval/PLAN.md).
- **Change:** none (harness validation; code at commit 1e549be + EXP-1 amendment commit).
- **Lane & model:** fireworks `deepseek-v4-pro` (strongest granted model; smoke-tested
  chat+json OK) + embeddings `qwen3-embedding-8b` via Fireworks (4096-dim, smoke-tested).
  New models granted by owner 2026-07-06: `nemotron-3-ultra-nvfp4`, `deepseek-v4-pro`,
  `deepseek-v4-flash` — all three pass chat+json smoke through the frozen client.
  (`qwen3-reranker-8b` is a reranker, not an embeddings endpoint — not usable via
  `get_embeddings`.)
- **Slice & control:** 5 knowledge-update questions, oracle split (`--max-questions 5`).
  No control needed (validation run, not a comparison). Nemotron rerun later = the
  reportable number.
- **Success bar (B1-0 gate):** all 5 questions complete with `error: None`, non-empty
  hypotheses, ≥1 substring hit; every graph + answer hand-read. NOT a quality
  measurement (n=5).
- **Cost estimate:** ~150–400 Fireworks calls, ~20–60 min.
- STATUS: RUNNING — `evaluation/DocRED/../LongMemEval/../../evaluation/LongMemEval` via
  `evaluation/LongMemEval/exp2_smoke_fw.sh`, log `evaluation/results/exp2_lme_smoke.log`,
  cache `evaluation/results/lme_kg_cache_fw_smoke/`, marker `EXP2_DONE`.

---

## PROCESS: anti-memorization guards codified  (2026-07-06, owner's order)
- Owner directive: the loop must do incremental experiment-driven updates (the pattern
  used throughout the DocRED arc), never brute-force score-chasing; the system must not
  MEMORIZE benchmarks (especially LongMemEval) — it must learn to adapt to information
  generally.
- Added SKILL.md §6 "Anti-memorization guards" (7 rules: dev/held-out on every benchmark,
  one structural change per experiment, no benchmark vocab in system code, immovable
  controls, cross-ability validation, metric triangulation, lessons-store-patterns-only).
- Pre-registered LongMemEval dev/held-out split in PLAN.md: per question-type indices
  0–19 dev / 20+ held-out; EXP-2's smoke questions (0–4) permanently development data.
- No system/eval code changed; process docs only.
