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

### EXP-1 verdict  (2026-07-06 23:15)
- **Result (glm-5p2, slice A, n=5):**
  | metric | glm singlepass | glm hybrid | nemotron singlepass (v4) | nemotron hybrid (v4/v5) |
  |---|---|---|---|---|
  | entR | **0.908** | **0.870** | 0.81 | 0.79 |
  | entP | 0.855 | 0.881 | ~0.86 | ~0.87 |
  | pairR | 0.329 | 0.323 | ~0.26 | ~0.28 |
  | relF1@0.6 | 0.261 | **0.289** | ~0.22 | ~0.21 |
  Zero-pair docs: none. Direction flips: 1 per strategy. Hybrid wall 7–11 min/doc,
  NO reasoning-runaway pathology (Nemotron doc-1 took far longer with ladder rescues).
- **Verdict: MODEL-BOUND signal (bars hit: hybrid entR 0.87 ≥ 0.85 ✗→borderline,
  relF1 0.289 ≥ 0.25 ✓).** Honest read at n=5: a stronger open-weight model lifts every
  layer (+0.08–0.10 entR, +0.07 relF1) and eliminates the runaway pathology on these docs.
  Two structural findings survive the model swap: (1) hybrid > singlepass on relF1 on
  BOTH models — the deliberation stack's value is real, not a Nemotron artifact;
  (2) hybrid entR still trails singlepass by ~0.04 on glm too — the G5 recall gap is
  PIPELINE-side (funnel/coref), not model-side. G3's zero-triple funnel did not reproduce
  on slice A; needs the actual doc-103 rerun after Phase 4 frees the comparison.
- **Action:** no code change (validation experiment). Informs: G5 stays top priority
  post-freeze; G2 answered at n=5 (headroom exists but pipeline gaps persist cross-model);
  local-lane reproduction impossible by definition (model IS the variable) — treat as
  lane evidence per SKILL.md §5. Follow-up: EXP-3 candidate = doc-103 funnel probe on glm.
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

---

### EXP-2 interim verdict: B1-0 gate FAILED — major robustness bug found  (2026-07-07 04:15)
- **Result (q0, deepseek-v4-pro, 5.25h build):** stages 1–4 extracted **654 entities
  (conf 0.90) + 1,727 triples + 795 wide-harvest seeds + 305 gleaned** on the
  conversational context — extraction WORKS on chat transcripts. Then evidence-linking
  batch 1 hit repeated Fireworks request timeouts (8 retries exhausted) →
  `process_document` raised → `process_corpus` swallowed it via
  `continue_on_document_error` (deliberative_orchestrator.py:1699–1717) → stage-9
  governance commit NEVER RAN → **final KG: 0 entities / 0 triples, runner `error: None`**
  → QA abstained. The founding "122→9" family reproduced as **186 calls → 0**, silently.
- **Verdict: the smoke did its job — requirement #1 violation confirmed on a NEW path.**
  A post-extraction enrichment-stage failure destroys all completed extraction work AND
  reports success. Model-agnostic bug (any transient outage triggers it); Fireworks
  timeout was just the trigger. This outranks every backlog item: new **G0**.
- **Action (eval-side now, freeze respected):** runner hardened — records
  `SUSPECTED_SILENT_PIPELINE_FAILURE` when 0 entities committed after >20 LLM calls.
  System-side fix pre-registered as EXP-3 (post-freeze). Pro run killed (q0 cached);
  rerun on flash = EXP-2b.

## EXP-2b: B1-0 smoke rerun on deepseek-v4-flash  (2026-07-07)
- **Hypothesis:** flash's much lower per-call latency avoids the timeout trigger and
  completes all 5 questions in ~1–2h total, giving the full B1-0 hand-read set; q0 on
  flash also tests whether the q0 wipeout is timing-dependent (flaky robustness).
- **Change:** eval-side only: silent-failure guard in LME runner (this commit); model
  pro→flash; fresh cache dir (no model mixing).
- **Lane & model:** fireworks `deepseek-v4-flash` + `qwen3-embedding-8b`.
- **Slice & control:** same 5 dev knowledge-update questions (permanently dev data).
- **Success bar:** B1-0 gate as pre-registered in EXP-2, plus: zero
  SUSPECTED_SILENT_PIPELINE_FAILURE records.
- **Cost estimate:** ~600–900 flash calls, ~1–2h.
- STATUS: RUNNING — `evaluation/LongMemEval/exp2b_smoke_flash.sh`, log
  `evaluation/results/exp2b_lme_smoke.log`, cache
  `evaluation/results/lme_kg_cache_fw_smoke_flash/`, marker `EXP2B_DONE`.

## EXP-3 (pre-registered, BLOCKED until Phase 4 frees multi_agent_kg/): graceful stage degradation
- **Hypothesis:** post-extraction enrichment failures (evidence linking, connectivity,
  deliberation-support stages) must degrade gracefully: log the stage as skipped-failed,
  proceed to stage 9 with what exists, commit the partial KG, and surface
  `failed_documents`/`degraded_stages` to callers. Preserves requirement #1.
- **Change (planned):** `process_document` per-stage try/except for enrichment stages;
  `process_corpus` exception path salvages staged results before counting the doc failed;
  MAB adapter + runners read `failed_documents` into their error fields.
- **Slice & control:** fault-injection test (mock evidence-linking exception) + q0 rerun;
  DocRED slice B regression (numbers must not move — the change only affects failure paths).
- **Success bar:** injected-failure doc commits >0 entities with error surfaced; 235
  tests + new fault-injection test pass; slice B scores unchanged.
