# Governed Knowledge Graph (multi-agent-kg)

## Project Overview
This project implements a **Governed Knowledge Graph**, which combines a standard knowledge graph with an explicit governance layer. In this architecture, domain expert agents own specific subgraphs, review proposed updates, and maintain an auditable structure.

The core contribution is treating information as governed subgraphs rather than isolated triples, using a multi-agent system to handle extraction, validation, and governance.

### Architecture
The extraction pipeline consists of a tiered multi-agent system coordinated by a `DeliberativeOrchestrator`:

1.  **Extraction Pipeline (Workers):**
    *   `DocumentProcessor`: Segments documents into manageable chunks.
    *   `DomainClassifier`: Infers domain and schema (entities/relations).
    *   `EntityExtractor`: Multi-stage extraction (initial, boundary, type, coreference).
    *   `RelationExtractor`: RHF-style relation extraction with open-world support.
    *   `EvidenceLinker`: Links triples to source sentences.

2.  **Validation & Integration (Coordinators):**
    *   `ExtractionValidator`: Iterative refinement and deliberation coordination.
    *   `ExtractionVerificationAgent`: Anti-hallucination checks against source text.
    *   `KnowledgeOrganizer`: Deduping, normalization, and integration into the KG.

### Core Components (`multi_agent_kg/core/`)
*   `GovernedKnowledgeGraph`: Wraps a `KnowledgeGraph` with governance semantics (propose/commit).
*   `Governance`: `OrgChart`, `Domain`, and assignment routing logic.
*   `DeliberativeOrchestrator`: Manages the end-to-end pipeline and agent communication.
*   `Memory`: `SharedMemory` (episodic, semantic, working, procedural) and `Blackboard` for deliberation.
*   `Communication`: `MessageBus` for inter-agent coordination.

## Building and Running

### Setup
```bash
# Install in editable mode
pip install -e .

# Environment setup
cp .env.example .env
# Set LLM_BACKEND (ollama or openai) and related keys
```

### Key Commands
*   **Run Pipeline:** `python scripts/run_pipeline.py --input <file.txt>`
*   **Build from SciERC:** `python scripts/build_governed_scierc.py --split dev --max-docs 10 --fixed-schema`
*   **Run QA Demo:** `python scripts/run_demo.py`
*   **Start QA Server:** `python scripts/qa_server.py`
*   **Run Tests:** `pytest tests/`
*   **Linting:** `black multi_agent_kg/ tests/` (100 char line length)
*   **Type Checking:** `mypy multi_agent_kg/ --python-version 3.11`

## Development Conventions

### Agent Development
*   All agents must inherit from `BaseAgent` (`agents/base.py`).
*   Agents should declare a `ModelTier` (SMALL, MEDIUM, LARGE) which resolves to specific models via the LLM client.
*   Confidence scores are floats [0, 1]. Deliberation is typically triggered in the [0.35, 0.65] range.

### Governance Modes
*   `strict`: Unknown triples escalate for human review.
*   `permissive`: All triples are auto-approved.
*   `audit_only`: Triples are auto-approved but every decision is logged in the audit trail (default for most runs).

### Deliberation Protocol
*   Workers submit hypotheses to the `Blackboard`.
*   The `DeliberationCoordinator` facilitates voting and debate among agents.
*   Weighted consensus determines the final decision.

## Current Priorities / Known Issues
*   **Orphan Node Rate:** Current builds show a high rate of orphan entities (~90%). A detailed fix plan involving fuzzy matching in `KnowledgeOrganizer`, removing entity caps in `RelationExtractor`, and adding an `OrphanLinker` agent is documented in `HANDOFF.md`.
*   **Missing Agents:** `EntityResolver`, `CriticAgent`, and `CorrectorAgent` are referenced in some docs but are not currently present in `multi_agent_kg/agents/`.
