# Agentic Graph Memory — Governed Knowledge Graphs as Agent Memory

A multi-agent system that builds **Governed Knowledge Graphs**: extraction is only
step one — every candidate fact is routed to a domain owner, reviewed against
evidence, admitted through an explicit governance decision, and stored with audit
metadata and provenance (including source dates, so newer facts supersede stale
ones). QA runs over domain-owned subgraphs rather than undifferentiated retrieval.

The system is **domain-agnostic by design**: schemas are discovered per corpus by a
domain classifier (no fixed general-purpose extractors), and every improvement is a
pre-registered, structurally-motivated experiment — never a benchmark-specific patch.

## Core idea

We treat information as governed subgraphs instead of isolated triples.

**Definition 1.** A Governed Knowledge Graph is a tuple `G = (E, T, D, φ, γ)` where:
- `E` is the set of entities
- `T ⊆ E × R × E` is the set of triples
- `D = {d₁, …, dₖ}` is the set of governed domains
- `φ: E → 2^D` maps each entity to one or more owning domains
- `γ` routes a proposed triple update to the responsible domain expert decision:
  `approve`, `reject`, `revise`, `escalate`, or `auto_approve`

Implementation: `multi_agent_kg/core/governed_kg.py`, `multi_agent_kg/core/governance.py`.

## Headline results

All numbers come from pre-registered experiments with pre-committed success bars;
the full trail (hypotheses, bars, misses, and reverts included) is in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

### Governance improves the admitted graph (SciERC test split, 100 docs, fixed schema)

Same protocol as the paper ("From Extraction to Governed Memory"), re-run entirely
on a **local 30B model** (Qwen3-30B-A3B on 2× L40S) at zero API cost:

| condition | triples | strict triple F1 | mapped triple F1 |
|---|---|---|---|
| flat insertion (Qwen3, local) | 3534 | 0.060 | 0.130 |
| **MAGG governed (Qwen3, local)** | 2389 | **0.080 (+33%)** | **0.152 (+17%)** |
| flat insertion (GPT-5, paper) | 2881 | 0.106 | 0.192 |
| MAGG governed (GPT-5, paper) | 1077 | 0.156 (+47%) | 0.290 (+51%) |

The governance delta is model-controlled within each row-pair; absolute GPT-5 vs
Qwen3 differences mix model and code. The paper build cost ~8M GPT-5 tokens per
100 docs; the local build costs $0. (EXP-PAPER-COMPARE, 2026-07-13)

### Extraction (DocRED, held-out docs)

| config | entity R | entity P | relF1@0.6 | pair R | median wall |
|---|---|---|---|---|---|
| singlepass (production extractor) | 0.88–0.91 | ~0.73 | ~0.14 | 0.21 | **~8 s/doc** |
| governed singlepass (opt-in, EXP-SPGOV-3) | 0.758 | **0.808** | **0.161** | 0.242 | 99 s/doc |
| + pair-completion stage (opt-in, EXP-PAIR-COMPLETE) | — | — | — | **0.305** | — |

The governed-singlepass trajectory (0.726 → 0.739 → 0.758 entR across three runs)
is the direct product of three shipped fixes: entity types preserved to the
organizer (GB-11), a numeric/date literal merge guard (GB-12: worst doc
0.462 → 0.808 entR), and parallel batch fan-out (GB-4: −38% wall on the API lane,
−20% local).

### Memory freshness (LongMemEval, knowledge-update)

Source-document dates thread end-to-end into triple provenance; the conflict
resolver supersedes stale facts on recency. Per-session dated ingestion moved
supersede events from ~0 to 1–20 per question and produced the system's first
knowledge-update gate pass. Corpora without dates see zero behavior change.

### Robustness

- **286 tests**; graceful stage degradation — a failing stage degrades and counts
  its drops instead of silently wiping the document (validated live: a previously
  twice-wiped question now commits 506 entities).
- **Embedding failover**: if the primary embedding endpoint exhausts its retry
  ladder, the process fails over to the Fireworks embeddings API automatically
  (validated in production: an infra wedge that previously killed a full run cost
  one document's wall time).

## Architecture

```
Document → [1 DocProcessor] → [2 DomainClassifier + Governance Bootstrap]
         → [3 EntityExtractor (deliberative | wide) + coref + provisional ownership]
         → [4 RelationExtractor | wide-harvest triples]
         → [4b Connectivity Pass] → [4c Pair Completion (opt-in)]
         → [5 EvidenceLinker] → [6 Deliberation] → [8 Verification]
         → [9 KnowledgeOrganizer: guarded dedup → propose_triple → governance → commit]
         → GovernedKnowledgeGraph (audit log, provenance, document dates, aliases)

GovernedKnowledgeGraph → domain-routed QA / incremental enrichment
```

Stage-9 dedup carries three domain-general guards: merge-into-aliases (no surface
form is ever lost), type compatibility (entities only merge into same-typed
canonicals; blank types permissive), and a literal guard (distinct numbers/dates
never merge, whatever the similarity score says).

### Extraction modes

| mode | pipeline | use |
|---|---|---|
| `deliberative` | full multi-stage + RHF | default for governed builds |
| `wide` (hybrid) | singlepass harvest front-end + full back-end | governance experiments |
| `governed_singlepass` (opt-in) | harvest → verify → govern (skips RHF/evidence/deliberation) | precision-leaning governed graphs at ~½ hybrid cost |
| singlepass | one call per doc (eval harness) | production raw extraction |

Opt-in switches (all default-off; controls never move):
`enable_pair_completion` (stage 4c), `LLM_BATCH_CONCURRENCY=N` (parallel batch
fan-out for verification/coref/evidence/connectivity), `EMBEDDING_FALLBACK_*`
(embedding failover — auto-armed when `FIREWORKS_API_KEY` is set).

## Compute setup

- **Production lane (all reportable numbers):** vLLM serving
  `Qwen/Qwen3-30B-A3B-Instruct-2507` (FP8) on 2× L40S; embeddings
  `mxbai-embed-large` via Ollama.
- **Experiment lane:** Fireworks API (`deepseek-v4-flash` fast/cheap,
  `glm-5p2` quality, `qwen3-embedding-8b` embeddings) — routed purely by env
  overrides, no code change. Wins found there must be reproduced locally before
  adoption.

## Quick start

```bash
pip install -e .
cp .env.example .env   # point LLM_BACKEND/VLLM_BASE_URL/EMBEDDING_BASE_URL at your stack

# Build a governed KG from text
python scripts/run_pipeline.py

# Build a governed KG from SciERC (paper protocol)
python scripts/build_governed_scierc.py --split test --max-docs 100 --fixed-schema

# DocRED extraction benchmark
python evaluation/DocRED/run_eval.py --strategy spgov --offset 100 --max-docs 20 \
  --save-kg-dir out_cache --output out.json
python evaluation/DocRED/score_docred.py --kg-dir out_cache --strategy spgov

# LongMemEval (memory abilities)
python evaluation/LongMemEval/run_eval.py --question-type knowledge-update --max-questions 5 \
  --save-kg-dir ckpt --output out.json
```

A React + Vite + d3 web UI (graph explorer, QA chat, governance review) lives on
the `feat/web-ui` branch (`frontend/app/` + `scripts/api_server.py`).

## Experiment discipline

Every change to the system follows the same loop (see
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for ~20 completed experiments):

1. **Pre-register before running**: hypothesis, exact change, slice, controls, and
   numeric success bars are committed before any result exists.
2. **One structural change per experiment** — changes must name the domain-general
   mechanism they fix; "makes the score go up" is rejected by construction.
3. **Dev/held-out splits everywhere; controls must not move**; no benchmark
   vocabulary anywhere in `multi_agent_kg/`.
4. **Honest verdicts**: misses are recorded as misses. Several of the most valuable
   findings came from REVERTed experiments (the SP-GOV reverts exposed two
   systemic dedup bugs that improved every mode).

### Current experiment plan (2026-07-14)

| next | goal | mechanism |
|---|---|---|
| 1. GB-3b | pair recall 0.305 → toward the 0.59 ceiling | structural evidence grounding: pair-completion triples must quote a literal substring of the document; non-quoting triples dropped in code (no reliance on instruction-following) |
| 2. GB-14 | close the review-quality gap vs the paper (strict P 0.051 vs 0.129) | design-first: evidence-grounded verification / LARGE-tier review / model headroom (gpt-oss-120b) — confidence thresholding already ruled out (F1 flat along the frontier) |
| 3. GB-5 | per-model JSON brittleness + schema leaks | vLLM structured decoding (enum-constrained types/relations) |
| 4. GB-2d | value-type collisions (differently-named relations, same fact) | domain-general object-shape collision detector — design before code |
| 5. LongMemEval breadth | 1 of 6 abilities validated | temporal-reasoning + multi-session slices on the fixed stack |
| parked | GB-6 verifier genre mismatch, GB-7 abstention, GB-13 schema hygiene | after the above |

Closed: GB-1 (robustness), GB-2/2b (freshness), GB-4 (latency), GB-8/11/12 (dedup
integrity), GB-9 (governed singlepass — opt-in, REVERT ×2 as default), GB-10 (QA
fallback crash).

## Project structure

```
multi_agent_kg/              # core package
  agents/                    # pipeline agents (extractor, verifier, organizer, …)
  core/                      # governed KG, governance, orchestrator, conflict resolution,
                             #   provenance, parallel fan-out, application layers
  llm/                       # LLM client (vLLM / OpenAI-compatible / Ollama) + embedding failover
evaluation/
  DocRED/                    # extraction benchmark harness + scorer + analyses
  LongMemEval/               # memory-ability benchmark harness
  kgafe/                     # QA answer-faithfulness evaluation
  governance/                # governed-update benchmarks
  adapters/ datasets/        # dataset adapters and data
scripts/                     # entry points (pipeline, SciERC builds, QA server, API server)
tests/                       # 286 tests incl. fault-injection suites for every shipped guard
EXPERIMENT_LOG.md            # the full pre-registered experiment trail (source of truth)
ROADMAP.md                   # system history and phase ladder
```
