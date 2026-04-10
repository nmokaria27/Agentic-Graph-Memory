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

# Spin up the QA server for the interactive explorer
python scripts/qa_server.py
```

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
  qa_server.py               # HTTP QA server
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
