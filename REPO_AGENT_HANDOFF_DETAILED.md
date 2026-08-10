# Agentic-Graph-Memory — Detailed Repository Handoff for Another Agent

## 1) Repository Identity

- **Repo:** `nmokaria27/Agentic-Graph-Memory`
- **Primary package:** `multi_agent_kg`
- **Project type:** Python 3.11+ multi-agent knowledge graph system with optional web UI
- **Core theme:** Build a **Governed Knowledge Graph** from unstructured documents, then run QA over domain-routed subgraphs.

## 2) Core Technical Thesis

The system does not treat extraction as “done after triples are generated.”  
It treats extraction as input to a governance pipeline:

- candidate entities/relations are extracted,
- evidence is linked,
- uncertain items can go through deliberation,
- updates are admitted through governance decisions,
- and provenance/audit metadata is preserved.

The README defines the governed KG abstraction as:
- entities `E`
- triples `T`
- governed domains `D`
- ownership mapping `φ`
- governance routing/decision function `γ`

Core governance implementation files:
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/governed_kg.py`
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/governance.py`

## 3) Stack Summary

### Backend / Core
- **Language:** Python
- **Packaging:** setuptools + `pyproject.toml`
- **Min Python:** `>=3.11`
- **Primary deps (root):**
  - `openai`
  - `python-dotenv`
  - `pydantic`
  - `numpy`
- **Optional deps:**
  - `pytest`, `black`, `mypy` (dev)
  - `optuna` (tuning)

### LLM + Embeddings Integration
From `multi_agent_kg/llm/openai_client.py` and `.env.example`:

- Chat backends: `ollama` (default), `vllm`, `openai`
- Fireworks-hosted model routing is supported per-model
- Embeddings can use a separate endpoint (`EMBEDDING_BASE_URL`)
- Retry/backoff/health-check controls are environment-driven

### Frontend
- **Path:** `frontend/app`
- **Framework:** React + Vite
- **Graph viz:** d3
- **Other notable dependency:** `react-markdown`

## 4) High-Level Runtime Architecture

Pipeline orchestrator:
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/deliberative_orchestrator.py`

Documented stage flow (README + AGENT_SCHEMA):
1. Document processing / segmentation
2. Domain classification + schema bootstrap
3. Entity extraction
4. Relation extraction
5. Evidence linking
6. Deliberation (when enabled)
7. Extraction validation
8. Extraction verification
9. Knowledge organization and graph admission

Supporting subsystems used by orchestrator:
- Shared memory (`multi_agent_kg/core/memory.py`)
- Message bus (`multi_agent_kg/core/communication.py`)
- Deliberation coordinator (`multi_agent_kg/core/deliberation.py`)
- Conflict resolution (`multi_agent_kg/core/conflict_resolution.py`)
- Provenance and KG operations utilities in `multi_agent_kg/core/`

## 5) Important Modes and Behavior Flags

The orchestrator supports multiple extraction modes (see README and orchestrator docstrings):

- `deliberative` (full staged flow)
- `wide` (hybrid/wide-harvest style front-end with backend processing)
- `governed_singlepass` (special governed singlepass path)

Useful behavior toggles (env/args seen across docs/code):
- Governance mode (`strict`, `permissive`, `audit_only`, and docs also mention `triage`)
- Pair completion enablement
- Deliberation enable/disable
- Checkpoint resume behavior
- Retrieval mode (`hybrid` default in config, plus lexical/dense/graph modes)

## 6) Entry Points (Operational)

### Build KG from text
- Script: `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/scripts/run_pipeline.py`
- Key runtime features:
  - accepts single `.txt` file or directory of `.txt` files
  - checkpoint discovery / resume controls
  - optional chunk-size and chunk-overlap
  - orchestration + export path

### API server + UI serving
- Script: `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/scripts/api_server.py`
- Provides endpoints such as:
  - `GET /health`
  - `GET /kg/data`
  - `GET /kg/stats`
  - `GET /models`
  - `POST /qa`
  - `POST /ingest`
  - `GET /pipeline/status`

### Frontend run/build
- `cd /home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/frontend/app`
- `npm install`
- `npm run dev` (local dev)
- `npm run build` (production build artifacts)

## 7) Evaluation / Benchmark Surface

Main evaluation tree:
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/evaluation`

Major benchmark and analysis areas visible in repo:
- `evaluation/DocRED`
- `evaluation/LongMemEval`
- `evaluation/Memory-Agent-Bench`
- `evaluation/kgafe`
- governance benchmark scripts and result summaries

There are many experiment logs and reports documenting iterative methodology:
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/EXPERIMENT_LOG.md`
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/ROADMAP.md`
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/LESSONS_AND_EXPERIMENTS.md`
- multiple benchmark-specific reports under `evaluation/`

## 8) Test & Quality Surface

- Tests directory: `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/tests`
- Current tree shows broad test coverage across:
  - governance flows
  - retrieval behavior
  - vector index behavior
  - extraction robustness/fallbacks
  - benchmark utilities/scorers

Typical commands (based on project tooling):
- `python -m pytest -q`
- `black .`
- `mypy multi_agent_kg`

## 9) Configuration and Environment

Primary config artifacts:
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/.env.example`
- `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/config.py`

Key env categories:
- backend selection (`LLM_BACKEND`)
- backend URLs and auth (`OLLAMA_BASE_URL`, `VLLM_BASE_URL`, API keys)
- model defaults (`LLM_DEFAULT_MODEL`, tier-specific model vars)
- retry/timeouts (`LLM_TIMEOUT`, retries/backoff)
- retrieval mode (`RETRIEVAL_MODE`)
- embedding settings (`EMBEDDING_*`)

## 10) Repository Layout (Practical Navigation)

Top-level areas that matter most for future agents:

- `multi_agent_kg/agents/`  
  Agent implementations (extractor, linker, validator, organizer, etc.)

- `multi_agent_kg/core/`  
  Orchestration, governance, graph model/ops, retrieval, QA, memory, checkpoints

- `multi_agent_kg/llm/`  
  LLM client abstraction + backend routing behavior

- `scripts/`  
  Operational entry points for pipeline, API server, benchmarking workflows

- `evaluation/`  
  Experiment harnesses, scoring, benchmark orchestration, analysis assets

- `tests/`  
  Regression and behavioral validation

- `frontend/app/`  
  React app for graph exploration + QA interface

## 11) Existing Design/Reference Docs Worth Reading First

Recommended order for an incoming agent:

1. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/README.md`
2. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/AGENT_SCHEMA.md`
3. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/ROADMAP.md`
4. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/EXPERIMENT_LOG.md`
5. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/deliberative_orchestrator.py`
6. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/core/governed_kg.py`
7. `/home/runner/work/Agentic-Graph-Memory/Agentic-Graph-Memory/multi_agent_kg/llm/openai_client.py`

## 12) Notes on Repo State and Documentation Style

- This repository already includes unusually rich internal documentation and experiment trail files.
- The README is not a lightweight intro only; it contains architecture, mode definitions, benchmark context, and run commands.
- Several docs preserve historical context and experiment chronology, useful for avoiding repeated dead-ends.

## 13) Short Handoff for Another Agent (Copy/Paste Ready)

If you only give one compact prompt to another agent, use this:

> Work in `multi_agent_kg` as the source of truth for core logic. Start with `README.md`, `AGENT_SCHEMA.md`, and `multi_agent_kg/core/deliberative_orchestrator.py`. This is a governed multi-agent KG pipeline (not plain triple extraction) with domain ownership, governance admission, deliberation, checkpointed runs, and QA/retrieval layers. Use scripts in `scripts/` for operational runs and `evaluation/` for benchmark workflows (DocRED, LongMemEval, Memory-Agent-Bench, KGAFE). LLM routing is backend-configurable (ollama/vllm/openai + optional Fireworks model routing) in `multi_agent_kg/llm/openai_client.py`; env knobs live in `.env.example` and `core/config.py`. Frontend is React/Vite in `frontend/app`, served alongside API endpoints from `scripts/api_server.py`.

