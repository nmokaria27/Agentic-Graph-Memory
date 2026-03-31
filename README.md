# Multi-Agent Knowledge Graph Framework

A multi-agent system that builds knowledge graphs from unstructured text and answers questions over them. Uses 9 specialized LLM agents that collaborate through shared memory, voting, and debate.

## What it does

**Extraction pipeline** -- Feed it a document, get back a structured knowledge graph:
- Domain classification and adaptive schema generation
- 4-stage entity extraction with coreference resolution
- Relation-Head-First (RHF) triple extraction with open-world discovery
- Evidence linking, multi-agent deliberation (voting + debate), anti-hallucination verification

**QA system** -- Ask questions over the extracted KG:
- Automatic domain clustering into expert agents with topic sub-agents
- Active graph exploration (iterative "do I need more info?" loops)
- Multi-expert debate when answers conflict
- Self-reflection critic that catches hallucinations before they reach the user
- Session memory across QA turns
- Full provenance chains mapping every claim back to KG triples

**Evaluation** -- Two evaluation frameworks:
- SciERC benchmark evaluation (entity/relation P/R/F1, type accuracy, hallucination rate)
- KGAFE (KG-grounded Atomic Fact Evaluation) -- novel framework that decomposes answers into atomic facts, verifies each against the KG via 3-tier verification, and runs a judge panel

## Quick start

```bash
pip install -e .

# Extract a KG from text
python scripts/run_pipeline.py

# Run the full demo (QA + KGAFE evaluation)
python scripts/run_demo.py

# Spin up the QA server for the interactive explorer
python scripts/qa_server.py
```

## Project structure

```
multi_agent_kg/              # core package
  agents/                    # 9 extraction pipeline agents
  core/                      # orchestrator, QA system, KG data structures, memory
  llm/                       # LLM client (Ollama / OpenAI)
  utils/                     # visualizer, debug logger
evaluation/                  # evaluation framework
  kgafe/                     # KG-grounded atomic fact evaluation
  adapters/                  # dataset adapters (SciERC)
  datasets/                  # benchmark data
scripts/                     # entry points
  run_pipeline.py            # extract KG from text
  run_demo.py                # full demo: QA + KGAFE eval
  run_full_test.py           # incremental enrichment + QA
  qa_server.py               # HTTP QA server
  extract_pdf.py             # PDF text extraction utility
```

## Architecture

```
Document --> [DocProcessor] --> [DomainClassifier] --> [EntityExtractor] --> [RelationExtractor]
    --> [EvidenceLinker] --> [Deliberation: voting + debate] --> [Validator] --> [Verifier]
    --> [KnowledgeOrganizer] --> Knowledge Graph

Question --> [Decompose + Route] --> [Active Explorer Experts] --> [Debate Arena]
    --> [Synthesizer] --> [Critic Agent] --> [Provenance Tracker] --> Answer
```

All agents share memory (episodic/semantic/procedural), a blackboard for hypothesis voting, and a message bus for inter-agent communication.

## Evaluation

```bash
# Run SciERC evaluation
python evaluation/run_evaluation.py --max-docs 5 --fixed-schema

# Run KGAFE evaluation on your QA system
python evaluation/kgafe/run_kgafe.py --kg kg_export.json --n-questions 20
```

## Config

Uses local Ollama LLMs by default (gemma3:27b). Set `LLM_BACKEND=openai` and `OPENAI_API_KEY` in `.env` to use OpenAI models instead.
