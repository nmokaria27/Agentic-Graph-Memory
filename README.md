# Governed Knowledge Graph

This repo now centers on a **GovernedKnowledgeGraph**: a knowledge graph plus an explicit governance layer where domain expert agents own subgraphs, review updates, and maintain an auditable domain structure. QA and enrichment remain in the repo, but they are application layers built on top of the governed data structure rather than the main research contribution.

## Core idea

We treat information as governed subgraphs instead of isolated triples.

**Definition 1.** A Governed Knowledge Graph is a tuple `G = (E, T, D, φ, γ)` where:
- `E` is the set of entities
- `T ⊆ E × R × E` is the set of triples
- `D = {d₁, …, dₖ}` is the set of governed domains
- `φ: E → 2^D` maps each entity to one or more owning domains
- `γ` routes a proposed triple update to the responsible domain expert decision:
  `approve`, `reject`, `revise`, `escalate`, or `auto_approve`

The implementation of this definition lives in:
- `multi_agent_kg/core/governed_kg.py`
- `multi_agent_kg/core/governance.py`

## What the repo does

**Governed KG creation**
- document ingestion through a multi-agent extraction pipeline
- preliminary domain bootstrap during creation from discovered schema
- provisional entity-to-domain assignment during extraction
- triple proposal flow through `propose_triple() -> governance routing -> commit_decision()`
- full audit log of governance decisions and ownership routing

**Application layers**
- QA over domain-owned subgraphs
- incremental enrichment of an existing governed KG
- governance benchmarks and extraction benchmarks

## Quick start

```bash
pip install -e .

# Build a governed KG from text
python scripts/run_pipeline.py

# Build a governed KG from SciERC
python scripts/build_governed_scierc.py --split dev --max-docs 10 --fixed-schema

# Run the application-layer demo (QA over an existing governed KG)
python scripts/run_demo.py

# Spin up the QA server for the interactive explorer (legacy, pairs with kg_explorer.html)
python scripts/qa_server.py

# Spin up the full API server (used by the frontend/ web app below)
python scripts/api_server.py --port 8000
```

## Frontend (Web UI)

`frontend/app/` is a React + Vite + d3 single-page app — graph explorer, QA chat with
streamed progress, governance review, document upload, and a live pipeline-progress
status bar. It talks to `scripts/api_server.py` over HTTP (see that file's docstring
for the full endpoint list). `evaluation/results/` (DocRED, etc. run caches) is also
browsable from the UI's "Graph source" dropdown, read-only.

**Dev mode** (hot reload, two processes):

```bash
python scripts/api_server.py --port 8000        # backend

cd frontend/app
npm install    # or: bun install
npm run dev    # or: bun run dev  ->  http://localhost:3000
```

The Vite dev server proxies `/api/*` to the backend (`frontend/app/vite.config.js`) —
adjust the `target` there if you run the API on a different port.

**Single-port deploy** (backend also serves the built frontend — useful behind an SSH
tunnel or a remote GPU box with no browser):

```bash
cd frontend/app
npm install && VITE_API_BASE='' npm run build   # or: bun install && VITE_API_BASE='' bun run build
cd ../..
python scripts/api_server.py --port 8000        # now also serves frontend/app/dist/ at /
```

Open `http://localhost:8000`. `VITE_API_BASE=''` makes the built app call the API on
its own origin instead of expecting the dev proxy.

No `node`/`npm` on the machine? [bun](https://bun.sh) is a drop-in replacement for both
commands above (`bun install`, `bun run build`, `bun run dev`).

Running on the UMD MindLabs cluster (gpu01/gpu02)? See `SERVER_GUIDE.md` §9.6 for the
exact port (vLLM occupies 8000 there — use `--port 5150`) and the SSH tunnel command to
view it from your laptop.

## Project structure

```
multi_agent_kg/              # core package
  agents/                    # 9 extraction pipeline agents
  core/                      # governed KG, governance, orchestrators, application layers
  llm/                       # LLM client (Ollama / OpenAI)
  utils/                     # visualizer, debug logger
evaluation/                  # evaluation framework
  kgafe/                     # QA application-layer evaluation
  governance/                # governed-update benchmarks
  adapters/                  # dataset adapters (SciERC)
  datasets/                  # benchmark data
scripts/                     # entry points
  run_pipeline.py            # create a governed KG from text
  build_governed_scierc.py   # build a governed KG from SciERC
  run_demo.py                # QA demo on top of a governed KG
  qa_server.py               # legacy HTTP QA server (pairs with kg_explorer.html)
  api_server.py              # full API server (used by frontend/app/)
frontend/app/                # React + Vite + d3 web UI — see "Frontend" above
```

## Architecture

```
Document --> [DocProcessor] --> [DomainClassifier]
         --> [Preliminary Domain Bootstrap]
         --> [EntityExtractor] --> [Provisional Entity Ownership]
         --> [RelationExtractor] --> [EvidenceLinker]
         --> [Verifier] --> [KnowledgeOrganizer]
         --> GovernedKnowledgeGraph

GovernedKnowledgeGraph --> [QA Application Layer] --> Answer
GovernedKnowledgeGraph --> [Enrichment Application Layer] --> Updated GovernedKnowledgeGraph
```

The extraction pipeline still uses shared memory, a blackboard, and a message bus, but the central object is now the governed data structure rather than the QA stack.

## Evaluation

```bash
# Extraction benchmark
python evaluation/run_evaluation.py --split dev --max-docs 10 --fixed-schema --reuse-corpus-schema

# Governance benchmark
python evaluation/governance/run_governance_benchmark.py \
  --kg-path evaluation/results/scierc_gold_governed_kg.json \
  --num-positive 40 --num-negative 40 --route-only \
  --output evaluation/results/governance.json
```

## Config

Uses local Ollama LLMs by default (gemma3:27b). Set `LLM_BACKEND=openai` and `OPENAI_API_KEY` in `.env` to use OpenAI models instead.
