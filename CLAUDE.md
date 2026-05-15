# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .

# Run tests
pytest tests/
pytest tests/test_governed_kg.py          # single test file

# Lint and type-check
black multi_agent_kg/ tests/              # 100 char line length
mypy multi_agent_kg/ --python-version 3.11

# Main pipeline entry points
python scripts/run_pipeline.py --input <file.txt> --governance-mode audit_only
python scripts/build_governed_scierc.py --split dev --max-docs 10 --fixed-schema
python scripts/run_demo.py                # QA demo with domain expert routing
python scripts/qa_server.py              # HTTP QA server

# Evaluation
python evaluation/run_evaluation.py --split dev --max-docs 10 --fixed-schema
python -m evaluation.kgafe.run_kgafe --kg-path kg_export.json --benchmark --n-questions 20
python evaluation/governance/run_governance_benchmark.py --kg-path <kg_path> --num-positive 40 --num-negative 40

# Verify LLM backend (Ollama)
curl localhost:11434/api/tags
```

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:

```bash
LLM_BACKEND=ollama             # or "openai"
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_DEFAULT_MODEL=gemma3:27b
LLM_SMALL_MODEL=qwen3:4b       # optional tier override
LLM_MEDIUM_MODEL=qwen3:8b      # optional tier override
LLM_LARGE_MODEL=gemma3:27b     # optional tier override
LLM_TIMEOUT=300
LLM_MAX_RETRIES=8
OPENAI_API_KEY=...             # only if LLM_BACKEND=openai
PIPELINE_GOVERNANCE_MODE=audit_only   # strict | permissive | audit_only
```

The default backend is Ollama (local or via SSH tunnel). The remote Ollama server is accessed through an SSH tunnel to `gpu01.mind.cs.umd.edu` via `mind-access00.cs.umd.edu`.

## Architecture

### Pipeline Flow (9 agents)

```
Document
  → DocumentProcessor        (chunk into ~1500-char segments)
  → DomainClassifier         (infer domain + entity/relation schema)
  → EntityExtractor          (4-stage: initial → boundary → type → coreference)
  → RelationExtractor        (RHF-style: identify → head-bind → tail-bind; open-world discovery)
  → EvidenceLinker           (link triples to source sentences, adjust confidence)
  ↓
  [DELIBERATION if 0.35 ≤ confidence < 0.65]
    Blackboard voting → conflict detection → debate → weighted consensus
  ↓
  → ExtractionValidator      (iterative refinement, up to 4 iterations, quality ≥ 0.85)
  → ExtractionVerificationAgent  (anti-hallucination check against source text)
  → KnowledgeOrganizer       (dedup, normalize, integrate into KG)
  ↓
GovernedKnowledgeGraph (OrgChart + domain assignments + audit log)
  ↓
Application layers: QA, incremental enrichment, KGAFE evaluation
```

### Core Infrastructure (`multi_agent_kg/core/`)

- **`deliberative_orchestrator.py`** — Main pipeline runner; instantiate `DeliberativeOrchestrator` to process a corpus.
- **`governed_kg.py`** — `GovernedKnowledgeGraph`: wraps `KnowledgeGraph` with governance layer (two-phase commit: propose → commit with approve/reject/revise/escalate action).
- **`governance.py`** — `OrgChart`, `Domain`, governance assignment routing.
- **`domain_experts.py`** — `DomainExpertAgent`, `QAOrchestrator`: domain-owned subgraph QA with routing.
- **`memory.py`** — `SharedMemory` with four stores: episodic (document context), semantic (cross-doc knowledge), working (current task), procedural (patterns); `Blackboard` for hypothesis posting and voting.
- **`communication.py`** — `MessageBus` and `CollaborationProtocol`; message types: INFORM, REQUEST, RESPONSE, PROPOSE, ACCEPT, REJECT, REFINE, DELEGATE, FEEDBACK.
- **`deliberation.py`** — `DeliberationCoordinator`; vote weights: EvidenceLinker 1.2, ExtractionValidator/VerificationAgent 1.5, DomainClassifier 0.8, DocumentProcessor 0.5; consensus threshold 0.6.
- **`knowledge_graph.py`** — `KnowledgeGraph`, `Entity`, `Triple`, `Conflict`; dedup via `_triple_set`.
- **`kg_operations.py`** — Load/save, diff, merge, entity matching.
- **`adaptive_config.py`** — Domain taxonomy, batching parameters, per-domain thresholds.
- **`incremental_enrichment.py`** — `IncrementalEnricher`, `ConflictResolver` for adding new documents to existing KGs.

### LLM Client (`multi_agent_kg/llm/openai_client.py`)

Single client abstraction compatible with both Ollama and OpenAI. Handles JSON parsing, retry logic, and tiered model selection (SMALL/MEDIUM/LARGE). All agents declare a `ModelTier`; the client resolves it to the configured model via environment variables.

### Governance Modes

| Mode | Behavior |
|---|---|
| `strict` | Unknown triples escalate for human review |
| `permissive` | All triples auto-approved |
| `audit_only` | Auto-approved but every decision logged |

### Evaluation

- **SciERC benchmark** (`evaluation/run_evaluation.py`): extraction P/R/F1, hallucination rate against 350/50/100 train/dev/test NLP paper dataset.
- **KGAFE** (`evaluation/kgafe/`): QA faithfulness — atomic decomposition → 3-tier verification (exact match → path-based → semantic) → multi-judge consensus panel.

## Key Conventions

- All agents inherit from `BaseAgent` (`agents/base.py`) and declare a `ModelTier`.
- Confidence scores are floats 0–1; deliberation triggers at [0.35, 0.65); conflicts at near-zero.
- The `relation_extractor.py` (85KB) and `knowledge_organizer.py` (43KB) are the largest and most complex agents.
- `AGENTS.md` in the repo root is the authoritative architecture reference (25KB). `AGENT_SCHEMA.md` covers agent data structure interfaces.
- Debug logging is optional; set `LLM_USAGE_LOG` to capture token usage per call.
