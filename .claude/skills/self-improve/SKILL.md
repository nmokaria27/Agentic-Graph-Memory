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
6. **Production inference is local** — gpu02 vLLM (2× L40S, 92 GB total), embeddings
   `mxbai-embed-large` on Ollama (gpu01). Local model as of 2026-07-07:
   **Qwen3-30B-A3B-Instruct-2507-FP8** (replaced Nemotron-30B after EXP-MODEL-LOCAL hit
   its bar; Nemotron restore recipe in SERVER_GUIDE §7.1). Fireworks (§5) is an
   *experiment lane*, not production: any win found there must be reproduced on the
   local stack before adoption.

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

**Worktree escape hatch (owner-approved 2026-07-07, first used for GB-8):** to port and
validate `multi_agent_kg/` changes WHILE the main tree is frozen, use a git worktree —
the live process never sees the edits. Checklist (each item bit us once):
1. `EnterWorktree` default-bases on `origin/main`, NOT your feature branch — immediately
   `git reset --hard feat/<branch>` inside the worktree and confirm the pytest count.
2. Bridge gitignored files manually: `cp <main>/.env .env`; symlink the DocRED data
   JSONs individually into `evaluation/DocRED/data/` (the dir has tracked files — don't
   symlink the whole dir).
3. Runs launched from the worktree may use gpu02 ONLY if the frozen run doesn't (e.g.
   it's on Fireworks) — the freeze rule protects code, not compute.
4. Commit on the worktree branch + push it (R2 holds), but MERGE into the feat branch
   only after the frozen run's done-marker — the merge rewrites the main tree's files.

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
- vLLM on gpu02 (2× L40S) serves **`Qwen/Qwen3-30B-A3B-Instruct-2507`** (FP8 MoE, 3B
  active — since 2026-07-07; `.env` `LLM_DEFAULT_MODEL` already points at it). Fast
  (~8 s/doc singlepass, ~1 s JSON calls) and non-thinking: `NEMOTRON_THINKING` /
  `NEMOTRON_BUDGET_CEILING` are inert for it. `VLLM_MAX_MODEL_LEN=131072`.
- Model swapping: one model at a time (TP=2 uses both GPUs); recipes for Qwen3 AND the
  previous production model (Nemotron-30B FP8 — needs `NEMOTRON_THINKING=on` + the
  truncation ladder envs) are in SERVER_GUIDE §7.1. Weights live in `/scratch/models/`.
  Candidates that fit 92 GB if needed: gpt-oss-120b (~63 GB, quality pick), Qwen3-32B
  dense. NVFP4 models do NOT run on L40S (Ada — no FP4 tensor cores).
- Embeddings: Ollama on gpu01 (`EMBEDDING_BASE_URL` in `.env`). Cheap; usable anytime.
- Use for: final/reportable numbers, anything feeding MATRIX_REPORT or phase verdicts.
- Do NOT run heavy local jobs while another local benchmark leg is measuring.
- All Nemotron-era baselines (matrix v1–v5, Phase 4, EXP-JUDGE-PHASE4) are cached in
  `evaluation/results/` and documented in MATRIX_REPORT — comparable via cached scores,
  no need to re-serve Nemotron unless re-extracting.

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
  (as of 2026-07: `kimi-k2p6`, `kimi-k2p5`, `glm-5p2`, `glm-5p1`, `gpt-oss-120b`,
  `nemotron-3-ultra-nvfp4`, `deepseek-v4-pro`, `deepseek-v4-flash`,
  `qwen3-embedding-8b`, `qwen3-reranker-8b`).
- **Model quirks (tested through the project client — trust this over assumptions):**
  `glm-5p2`, `gpt-oss-120b`, `deepseek-v4-pro/flash`, `nemotron-3-ultra` all pass
  chat+JSON. **`kimi-k2p6` BREAKS `chat_completion_json`** (nested-fragment extraction →
  silent 0-entity docs); `kimi-k2p5` 500s server-side. `qwen3-reranker-8b` is a
  reranker, NOT embeddings. Default experiment models:
  `deepseek-v4-flash` (fast/cheap), `glm-5p2` (quality), embeddings
  `qwen3-embedding-8b` (4096-dim) via `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL` overrides.
  deepseek-v4-pro is SLOW (1–2 min/call) — smoke gates only, never bulk.
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

## 7. Goal backlog — v2, re-derived 2026-07-07 after Phase 4 + smoke verdicts

Experiment naming: `EXP-<AREA>-<WHAT>` (descriptive; see the naming table at the top of
EXPERIMENT_LOG.md — old sequential ids remain as aliases).

| # | weak area (evidence) | goal-based operation | status |
|---|---|---|---|
| GB-1 | **Silent work loss** (3 wipeouts in 7 smoke runs — EXP-LMESMOKE-*) | graceful stage degradation + shape guards (**EXP-ROBUST-DEGRADE: SHIPPED, 240-test baseline**); **EXP-ROBUST-VALIDATE: PASSED end-to-end (5/5, 0 errors, 0 silent failures, q0/q4 — the two wiped questions — now build 506e/364e)** | **CLOSED** |
| GB-8 | **Organizer dedup over-merge = mass entity deletion** (Qwen3 hybrid slice A: 32 extracted → 2 in KG; the coref-collapse pathology recurring in `_deduplicate_entities`, which DELETES merged entities — coref's merge-never-delete guard (commit 60ca03e) was never applied here) | merge-into-aliases instead of delete; cap merge-group size; require type compatibility; fault-injection test with an aggressive-merge response. **EXP-DEDUP-GUARD: ACCEPT, merged `5f3642e`** — slice-A entR 0.244→0.801, dangling endpoints 27/29→2/258, 240-test baseline | **CLOSED** |
| GB-2/2b | **Stale answers on knowledge updates.** Root cause chain, fully traced: `find_conflicts()` was never broken; the resolver was always wired; the actual gap was (1) no document date reached triple provenance, and (2) LongMemEval concatenated all sessions into one undated blob | **EXP-FRESHNESS-QA + EXP-FRESHNESS-E2E: ACCEPT, merged.** `document_date` threaded end-to-end; per-session dated ingestion (`reuse_corpus_schema` + `enable_cross_document` for multi-doc). Supersedes: ~0 → 1-20/question on both lanes. **B1-0 gate PASSED for the first time** (local Qwen3, 1/5 substring, 2/5 hand-read correct — 3 prior runs were 0/5). q0 "25:50" and q2 "suburbs" now serve correctly on local | **CLOSED** |
| GB-10 | **QA crashes with `AttributeError: 'AdvancedQAOrchestrator' object has no attribute '_community_context'`** — found via GB-2b's local-leg run (3/5 empty answers); a borrowed `QAOrchestrator` fallback method needs it, latent since 0932808 | Added `_community_context` to `AdvancedQAOrchestrator`; AST-audited no other borrowed-method attr is missing; 2 regression tests. `requery_cached.py` re-runs just the QA step (~30s) on cached KGs instead of re-extracting (4h) | **CLOSED** |
| GB-2c | **Relation-name variance blinds conflict detection** — q0's KG held `HAS_TIME`/`HAS_VALUE` for the same fact ACTIVE simultaneously; exact-key `find_conflicts()` never compares them even with a perfect date signal | Embedding-similar-relation conflict candidates (mirrors `_relation_outside_schema`'s 0.85 tier). **Real-embedding smoke test: works for genuine relation synonyms** (spouse_of/married_to 0.88, ceo_of/chief_executive_officer_of 0.89) **but NOT the motivating case** (has_time/has_value: 0.69 — wrong signal, relation names aren't semantically close even when they name the same fact) | **PARTIAL — shipped, doesn't close q0** |
| GB-2d | **The actual q0 pattern**: an attribute-holder subject (e.g. `personal_best_time`) accumulates multiple active outgoing triples with different relations but value-typed objects (numbers/durations/dates) — GB-2c's relation-embedding signal can't catch this; needs object-type/shape co-occurrence per subject instead | Design a domain-general "value-type collision" detector (no relation vocabulary assumptions) — **pre-register the design BEFORE writing code** (GB-2c was implemented before its pre-registration, a process slip; don't repeat it) | blocked on design |
| GB-9 | **Governed singlepass (SP-GOV)** — implemented (`extraction_mode="governed_singlepass"`, `--strategy spgov`, 259-test baseline). **EXP-SPGOV: REVERT as default** — entP +0.076 PASS but entR 0.726 vs 0.828 bar (organizer same-type over-merge on list-heavy docs: doc_109 29→9 entities, "Dr. Beat" absorbed 5 distinct albums), relF1 and wall bars also missed. Mode stays in-tree, opt-in | **EXP-SPGOV-2 (post-GB-11): REVERT ×2** — entR 0.739 (bar 0.828, MISS), entP 0.764 PASS, relF1@0.6 0.137 at-bar PASS, wall 123.9s MISS. GB-11 validated live (doc_109 9→21 entities, cross-type merges gone); residual = same-type over-merge (doc_100: 16/37 merged; 1911/1992 absorbed by 1939, all `DATE`) + wide-harvest run variance. Re-open blocked on GB-12 + GB-4 (wall) | **CLOSED (REVERT ×2, opt-in retained)** |
| GB-11 | **Entity types lost before stage 9 / dumps** — two bugs: (1) coref defaulted missing LLM types to `UNKNOWN` (erased member extraction types; GB-8 saw every entity as same-typed); (2) eval dumps read nonexistent `Entity.entity_type` → always `'?'`. **EXP-TYPE-PROP: ACCEPT** — `inherit_type_from_members` in coref; blank-type set (`UNKNOWN`/`?`) in GB-8 gate + semantic dedup; dump reads `Entity.type`; 7 new fault-injection tests | shipped; unblocks EXP-SPGOV-2 | **CLOSED** |
| GB-12 | **Distinct numeric/date literals merge in stage-9 dedup** (doc_100: 1911/1992 absorbed by 1939 — same type `DATE`, size-legal group, so GB-8's gate is correctly permissive; same exposure in hybrid) | domain-general guard: entities whose surface forms are distinct numeric/date literals must never merge (pure string/shape check, no LLM, no dataset vocabulary); fault-injection test with a year-absorbing merge response | **NEXT (small)** |
| GB-3 | **Pair discovery is the binding constraint** (~23% of gold pairs found; relation naming is NOT the problem — EXP-JUDGE-PHASE4). Cross-lane evidence from EXP-FRESHNESS-E2E: Fireworks flash missed extracting q2's "suburbs" update entirely (0 entities for it) and modeled q0's update as a disjoint goal-entity instead of updating the existing one — a live example of the recall gap, on the Fireworks lane specifically (local Qwen3 got both right) | offline coverage analysis on Phase-4 caches, then 2-sample harvest union (#7) / targeted re-glean (#2) | after GB-9 |
| GB-4 | **Extraction cost/latency** (owner priority; largely relieved by Qwen3 — 8 s/doc singlepass) | remaining: parallelize per-batch LLM calls (asyncio); cap/skip evidence-linking for memory ingestion; EXP-LONGDOC-POSITION (recall-by-position curve, whole-doc vs segmented — owner's long-doc hypothesis) | with GB-9 |
| GB-5 | **Per-model JSON brittleness** (kimi fragment bug; dedup str crash; Qwen3 aggressive merges; `_THINKING_MODEL_PATTERNS` hardcoded) | vLLM guided/structured decoding (IDEAS #1) + `LLM_THINKING_MODELS` env override | after GB-2 |
| GB-6 | **Verifier genre mismatch** (1075/1997 triples "hallucinated" on conversational text) | hand-read 30 rejected triples from smoke caches; if real, genre-neutral verification framing | parked |
| GB-7 | **Abstention quality** (Phase B `_abs` questions) | retrieval-sufficiency gate, scored on `_abs` dev slice | Phase B1-1+ |
| retired | G6 direction post-check (0 genuine inversions in 80 runs); G2 headroom (answered: model-bound + pipeline gaps); SC re-validation folded into GB-5 | — | — |

After each major verdict: re-derive this table — retire done goals, add new failure
modes from hand-reads, re-rank by (expected gain × evidence) / effort.

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

## 9. State snapshot (2026-07-07 ~18:45 — trust EXPERIMENT_LOG.md for anything newer)

- **Phase A (DocRED) CLOSED.** Phase 4 n=40 verdict: hybrid v2 freeze OVERTURNED
  (lost entR on 29/40 docs at 30× cost); EXP-JUDGE-PHASE4 closed the decision —
  **singlepass = production extractor**. Hybrid retained only for Phase B
  governance/conflict experiments. Full detail: MATRIX_REPORT.md top section.
- **Local model = Qwen3-30B-A3B** (EXP-MODEL-LOCAL: entR 0.908 @ 8 s/doc, bar hit).
  Nemotron-era baselines live in cached scores; SERVER_GUIDE §7.1 has both recipes.
- **Robustness (GB-1):** graceful-degradation shipped (test baseline **237**);
  EXP-ROBUST-VALIDATE q0 gate PASSED (506 ents committed on the twice-wiped question);
  q1–q4 were running on the Fireworks lane at snapshot time — **check
  `evaluation/results/exp_robust_validate.log` for `EXPRV_DONE` and write the final
  verdict if nobody has** (watcher may have died with the session).
- **Known open bug (GB-8, do this first):** organizer LLM-dedup deletes merged
  entities; Qwen3's aggressive merge responses collapse 32→2. Blocks SP-GOV.
- **The thesis gap is confirmed 3× (GB-2):** KG stores updated facts but QA serves the
  stale one ("Chicago"/suburbs, "$350k"/"$400k", "27:12"/"25:50") — all on dev
  questions; freshness assembly is the highest-value mechanism next.
- **B1-0 gate not yet passed** (0/5 substring on the pre-fix run); rerun on fixed code
  + local Qwen3 becomes the real attempt after GB-8/GB-2.
- Owner's standing emphases this week: incremental pre-registered experiments (never
  score-chasing), anti-memorization guards (§6), doc-on-every-change + push-always
  (§4), Fireworks for parallel experiments, robustness above all.

## 10. Fresh-agent checklist

- [ ] Read §2 docs (30 min well spent; do not skip EXPERIMENT_LOG.md)
- [ ] `ps aux | grep run_eval` — know the freeze state before touching anything
- [ ] `git log --oneline -10` + `git status` — know the branch state
- [ ] `set -a && source .env && set +a` — env loaded (never print it)
- [ ] Verify lanes: local vLLM up? (`curl -s $VLLM_BASE_URL/models`) Fireworks key valid?
- [ ] Find the first `STATUS: RUNNING` entry in EXPERIMENT_LOG.md without a verdict —
      that experiment may have finished while nobody watched; score it FIRST.
