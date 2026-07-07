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
