---
name: self-improve
description: Self-improvement loop for the Agentic-Graph-Memory system — context bootstrap, experiment doctrine, the Fireworks parallel lane, and the logging/push discipline. Use when asked to improve the system, run/design experiments, work a goal from the improvement backlog, or continue the loop.
---

# Self-Improvement Loop — Agent Guide

You are improving a multi-agent governed knowledge-graph memory system. This skill tells
you how to load context, what you may and may not touch, how to run experiments on two
compute lanes, and the non-negotiable logging/push discipline. Follow it exactly.

## 1. Mission & the owner's standing requirements

Make the system measurably better at building and querying knowledge graphs as memory,
under these standing rules (never violate them to win a benchmark):

1. **Robust** — no silent mid-run failures, ever. Partial results are preserved, drops are
   counted and printed.
2. **Domain-adaptive** — no fixed general-purpose extractors (GLiNER rejected). Schema is
   discovered per corpus by `DomainClassifier`.
3. **Quality first** — rich, correct KG data beats speed.
4. **No benchmark bias** — fixes must be structural/domain-general. Guards: held-out
   slices, untouched controls, cross-benchmark checks. Nothing keyed to a dataset's
   vocabulary may enter `multi_agent_kg/`.
5. **Experiment → document → decide** — pre-register before running, measure against
   the bar, write the verdict honestly (misses reported as misses).
6. **Production inference is local** — Nemotron-30B FP8 on vLLM (gpu02), embeddings
   `mxbai-embed-large` on Ollama (gpu01). Fireworks (§5) is an *experiment lane*, not
   production: any win found there must be reproduced on the local stack before adoption.

## 2. Context bootstrap (do this first, in order)

1. `ROADMAP.md` — system description, full history, phase ladder, current constraints.
2. `EXPERIMENT_LOG.md` — what the loop has already tried; the LAST entries are the
   current state. Never repeat a settled experiment.
3. `evaluation/DocRED/MATRIX_REPORT.md` — measured extraction results v1→v5 (newest first).
4. `evaluation/DocRED/LESSONS.md` + `evaluation/DocRED/FLIP_ANALYSIS.md` — failure modes.
5. `RESEARCH_IDEAS.md` — 25 vetted techniques with experiment sketches; the goal backlog
   (§7) references these by number.
6. `evaluation/LongMemEval/PLAN.md` — Phase B doctrine.

Codebase questions: run `graphify query "<question>"` (from repo root) BEFORE reading or
grepping raw source; `graphify path "<A>" "<B>"` for relationships. Only read raw files
after graphify orients you. This applies to any subagent you spawn too — put it in their
prompts. Check `~/.claude/projects/-home-nmokaria-Agentic-Graph-Memory/memory/` for
session memories.

## 3. Check what is running before you act

A benchmark may be running from this working tree. Before ANY code edit:

```bash
ps aux | grep -E "run_eval|phase" | grep -v grep          # live benchmark processes?
tail -5 evaluation/results/*.log 2>/dev/null              # recent activity
```

**FROZEN-CODE RULE:** while any benchmark process launched from this working tree is
alive, do NOT edit `multi_agent_kg/` — a crash-restart would silently pick up your edits
mid-measurement. Everything in `evaluation/`, `tests/`, docs, and new files is safe.
When you need to prototype system-layer logic during a freeze, write it as a standalone
script under `evaluation/` operating on cached outputs, and port it after the freeze.

## 4. Non-negotiable discipline (the owner's explicit orders)

- **R1 — Log every change.** EVERY experiment and EVERY code change gets an entry in
  `EXPERIMENT_LOG.md` (template in §6): pre-registration BEFORE running, verdict AFTER.
  No exceptions, including failed/abandoned attempts — those are the most valuable entries.
- **R2 — Push periodically.** Commit after every completed experiment or accepted change;
  `git push origin feat/vector-index-and-qa-improvements` after every commit (or at
  minimum at the end of each loop iteration). A copy of the code must exist on GitHub at
  all times. `gh` lives at `~/miniconda3/bin/gh`.
- **R3 — Tests + graph.** After any code change: `python -m pytest -q` (baseline: 235
  passing — check EXPERIMENT_LOG.md for the current number) and `graphify update .`.
- **R4 — Secrets.** Credentials live in `.env` (gitignored). Load with
  `set -a && source .env && set +a`. NEVER print values; never commit them.
- **R5 — Results are gitignored.** `evaluation/results/` never reaches git — verdicts
  must be written into tracked docs (EXPERIMENT_LOG.md / MATRIX_REPORT.md).

## 5. Two compute lanes

### Local lane (production truth)
- vLLM on gpu02 serves `nvidia/nemotron-3-nano` (Nemotron-30B FP8). `NEMOTRON_THINKING=on`
  is REQUIRED (reasoning-off ⇒ blank extraction). `VLLM_MAX_MODEL_LEN=131072`,
  `NEMOTRON_BUDGET_CEILING=65536`. Truncation-retry ladder rescues reasoning runaway.
- Embeddings: Ollama on gpu01 (`EMBEDDING_BASE_URL` in `.env`). Cheap; usable anytime.
- Use for: final/reportable numbers, anything feeding MATRIX_REPORT or phase verdicts.
- Do NOT run heavy local jobs while another local benchmark leg is measuring.

### Fireworks lane (parallel experiments — no gpu02 contention)
The client (`multi_agent_kg/llm/openai_client.py`) routes by env, so ANY run can target
Fireworks per-process with no code change:

```bash
set -a && source .env && set +a
LLM_BACKEND=vllm \
VLLM_BASE_URL=https://api.fireworks.ai/inference/v1 \
VLLM_API_KEY=$FIREWORKS_API_KEY \
LLM_DEFAULT_MODEL=accounts/fireworks/models/<model> \
python -u evaluation/DocRED/run_eval.py ...
```

- List available models:
  `curl -s -H "Authorization: Bearer $FIREWORKS_API_KEY" https://api.fireworks.ai/inference/v1/models`
  (as of 2026-07: `kimi-k2p6`, `kimi-k2p5`, `glm-5p2`, `glm-5p1`, `gpt-oss-120b`).
  Default experiment model: `accounts/fireworks/models/kimi-k2p6` (strong, non-thinking).
- `NEMOTRON_*` env vars are inert for non-Nemotron model names.
- **Always use a fresh cache dir per (model × experiment)** — e.g.
  `evaluation/results/docred_kg_cache_fw_<tag>/`. Never mix models in one cache dir.
- Uses: pipeline/prompt/config experiments while gpu02 is busy; model-headroom
  comparisons; SC re-validation; structured-output (`response_format`) prototypes; cheap
  LLM-judge scoring; LongMemEval harness validation.
- Caveat: a Fireworks-lane win is *evidence*, not a result. Reproduce on the local lane
  before calling it an improvement (requirement #6). Fireworks results go in
  EXPERIMENT_LOG.md, never in MATRIX_REPORT.md.

## 6. Experiment doctrine

Slices: DocRED slice A = docs 0–4 (diagnostic, hand-readable), slice B = docs 30–34
(held-out, NEVER tuned on), Phase-4 docs 100–139 (fresh, n=40). LongMemEval: 5-question
smoke slices first (`--dry-run` to inspect for free). Singlepass = untouched control —
if a "measurement fix" changes the control's numbers, it's a bug or bias.

Never conclude from n=5 beyond "runs/doesn't"; rhf swings ±0.08 entR at n=5.

### Anti-memorization guards (owner's explicit order, 2026-07-06)

The loop improves the system's *mechanisms for adapting to information it sees* — it must
never make the system *memorize the benchmark*. Incremental, experiment-driven updates
only; no brute-force "try everything against the score" sweeps against a fixed question
set. Enforcement:

1. **Dev/held-out split on EVERY benchmark, always.** Any question/doc you hand-read,
   debug against, or iterate on is DEVELOPMENT data forever (like DocRED slice A). Each
   ability gets a frozen HELD-OUT subset that is scored at most once per accepted change
   and never inspected. LongMemEval: per question-type, filtered-list indices 0–19 = dev,
   20+ = held-out (pre-registered in evaluation/LongMemEval/PLAN.md). The smoke questions
   (indices 0–4) are permanently dev.
2. **One pre-registered structural change per experiment.** If you cannot state the
   domain-general mechanism a change fixes ("dates must survive ingestion"), it does not
   run. Changes phrased as "this makes the score go up" are rejected by construction.
3. **No benchmark vocabulary in `multi_agent_kg/`** — no question-type names, no
   dataset-specific formats, prompts, or heuristics. Session/context formatting lives in
   the eval adapters. Audit with grep before every accept (the DocRED bias-audit pattern).
4. **Controls must not move.** Keep an untouched control configuration per benchmark;
   if a change alters the control's numbers, it is a bug or bias — investigate, don't accept.
5. **Cross-validate accepted changes** on a second ability or benchmark before calling
   them wins (e.g., a knowledge-update fix must not regress temporal-reasoning; DocRED
   fixes get sanity-checked on the other benchmark's slices).
6. **Triangulate metrics** — substring + LLM-judge + hand-reads. Never optimize against
   the judge alone (Goodhart); a judge-only gain with flat substring/hand-read is
   INCONCLUSIVE, not ACCEPT.
7. **Lessons-as-memory (Phase D Tier 2) stores failure PATTERNS, never dataset content**
   — "biography docs: coref over-merges into the protagonist" is a lesson; a gold answer
   or question phrasing is contamination. Never store benchmark answers anywhere the
   system can retrieve them.

**Pre-registration template (append to EXPERIMENT_LOG.md BEFORE running):**

```markdown
## EXP-<n>: <name>  (<date>)
- **Hypothesis:** <what you believe and why — cite RESEARCH_IDEAS #N or a failure mode>
- **Change:** <exact code/config/prompt delta; commit of the base state>
- **Lane & model:** <local nemotron | fireworks <model>>
- **Slice & control:** <docs/questions; what stays untouched>
- **Success bar:** <pre-committed numeric threshold>
- **Cost estimate:** <calls/time>
- STATUS: RUNNING <cmd + log path + cache dir>
```

**Verdict (append when scored — never edit the pre-registration):**

```markdown
- **Result:** <numbers vs bar, per metric>
- **Verdict:** ACCEPT / REVERT / INCONCLUSIVE (+ why, honestly)
- **Action:** <merged as commit <sha> | reverted | follow-up EXP-<m>>
```

Run pattern (crash-safe): `nohup bash <script> > /dev/null 2>&1 &` with a done-marker
(`echo EXP_N_DONE >> log`), per-item checkpoints make reruns free (cache-skip). Scoring
is always offline: `score_docred.py` / `score_longmemeval.py` on the cache dir.

### Harness quick reference

```bash
# DocRED extraction run (any lane)
python -u evaluation/DocRED/run_eval.py --strategy hybrid|singlepass|rhf \
  --max-docs 5 --offset 0 --save-kg-dir <cache> --output <summary.json>
# Score (embeddings via gpu01; add --judge for LLM-judge relation scoring)
python evaluation/DocRED/score_docred.py --kg-dir <cache> --strategy <s> [--judge] --output <scores.json>
# Direction-flip mining (offline, zero LLM)
python evaluation/DocRED/analyze_flips.py --kg-dirs <cache...> --strategies <s...>
# LongMemEval (ability slice; --dry-run first)
python -u evaluation/LongMemEval/run_eval.py --question-type knowledge-update \
  --max-questions 5 --save-kg-dir <cache> --output <summary.json>
python evaluation/LongMemEval/score_longmemeval.py --cache-dir <cache> --output <scores.json> [--judge]
```

## 7. Goal backlog (work top-down unless the owner reprioritizes)

| # | goal | mechanism | lane now | ref |
|---|---|---|---|---|
| G1 | Validate LongMemEval harness end-to-end | B1-0 smoke (5 knowledge-update Qs) via Fireworks | FW now; Nemotron rerun for real numbers later | LME PLAN.md |
| G2 | Measure model headroom | slice A singlepass+hybrid on kimi-k2p6 vs Nemotron baselines | FW now | EXP-1 |
| G3 | Diagnose hybrid zero-triple funnel docs (doc 103 class) | rerun failing doc on FW model: reproduces ⇒ pipeline bug, else model-specific; then design re-glean trigger | FW now | IDEAS #2 |
| G4 | Retire empty-output failure class | guided-JSON / `response_format` prototype on FW; port to vLLM guided decoding after freeze | FW proto now | IDEAS #1 |
| G5 | Close hybrid entity-recall gap (~0.07 vs singlepass) | coref merge-gating, 2-sample wide-harvest union | FW proto; local port post-freeze | IDEAS #4/#5/#7 |
| G6 | Direction post-check | value-subject swap rule + typed-argument spot-check; prototype offline on caches | offline now; stage-9 port post-freeze | FLIP_ANALYSIS.md |
| G7 | Knowledge-update mechanics for B1 | deterministic freshness assembly; bi-temporal supersede | eval-side proto now; QA-layer port post-freeze | IDEAS #18, #16/#17 |
| G8 | SC re-validation | temp 0.4, explicit max_tokens, item-level union | FW now (cheap) | EXTRACTION_EXPERIMENTS.md |

After each Phase-4-style verdict lands, re-derive this table: retire done goals, add new
failure modes from hand-reads, re-rank by (expected gain × evidence) / effort.

## 8. The loop (one iteration)

1. Bootstrap context (§2); check running processes (§3).
2. Pick the top unblocked goal from §7 given the current freeze state and lanes.
3. Pre-register the experiment in EXPERIMENT_LOG.md (§6). Commit the pre-registration.
4. Run it (background + marker + checkpoints). While it runs, prep the next experiment
   or do offline analysis — never idle-wait on a long run.
5. Score offline; append the verdict with honest numbers.
6. ACCEPT ⇒ port/merge the change (respecting §3), run `pytest` + `graphify update .`,
   update any affected doc (ROADMAP running-state, MATRIX_REPORT if local-lane DocRED).
   REVERT ⇒ record why; the log entry IS the deliverable.
7. Commit + push (R2). One experiment = at least one commit.
8. Repeat, or stop and summarize if the owner's attention is needed (a standing
   requirement is at risk, a result contradicts prior verdicts, or budget/lane limits hit).

## 9. Fresh-agent checklist

- [ ] Read §2 docs (30 min well spent; do not skip EXPERIMENT_LOG.md)
- [ ] `ps aux | grep run_eval` — know the freeze state before touching anything
- [ ] `git log --oneline -10` + `git status` — know the branch state
- [ ] `set -a && source .env && set +a` — env loaded (never print it)
- [ ] Verify lanes: local vLLM up? (`curl -s $VLLM_BASE_URL/models`) Fireworks key valid?
- [ ] Find the first `STATUS: RUNNING` entry in EXPERIMENT_LOG.md without a verdict —
      that experiment may have finished while nobody watched; score it FIRST.
