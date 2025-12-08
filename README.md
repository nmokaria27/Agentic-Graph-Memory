# Multi-Agent Knowledge Graph Enrichment Framework

A deliberative multi-agent framework for building knowledge graphs using LLM-backed agents. The system features a tiered 8-agent architecture with specialized workers and coordinators, integrated with a novel shared memory and blackboard system for cross-document reasoning and collaborative deliberation.

## Architecture

### 8-Agent Pipeline

The system uses a tiered architecture inspired by KARMA research, with 5 worker agents for extraction and 3 coordinator agents for validation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DeliberativeOrchestrator                        │
│         Coordinates all agents with SharedMemory + MessageBus       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────┐
    │                 Shared Infrastructure                │
    │  SharedMemory: episodic, semantic, working memory   │
    │  Blackboard: hypothesis posting and voting          │
    │  MessageBus: inter-agent communication              │
    └─────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │              WORKER AGENTS                 │
         │                                            │
         │  [1] DocumentProcessor                     │
         │       ↓ segments                           │
         │  [2] DomainClassifier                      │
         │       ↓ domain config                      │
         │  [3] EntityExtractor (4-stage pipeline)    │
         │       ↓ entities                           │
         │  [4] RelationExtractor (RHF + open-world)  │
         │       ↓ triples                            │
         │  [5] EvidenceLinker                        │
         │       ↓ linked evidence                    │
         └────────────────────┬──────────────────────┘
                              │
         ┌────────────────────┴──────────────────────┐
         │           COORDINATOR AGENTS               │
         │                                            │
         │  [6] ExtractionValidator                   │
         │       ↓ validates + refines (max 4 iters)  │
         │  [7] ExtractionVerificationAgent           │
         │       ↓ anti-hallucination check           │
         │  [8] KnowledgeOrganizer                    │
         │       → integrates into KnowledgeGraph     │
         └────────────────────────────────────────────┘
```

### Worker Agents (Extraction)

| Agent | Role | Key Features |
|-------|------|--------------|
| **DocumentProcessor** | Ingests documents, segments into chunks | Semantic chunking, source tracking |
| **DomainClassifier** | Classifies into 6 domains | Domain-specific prompts and entity types |
| **EntityExtractor** | 4-stage entity extraction | Initial → Boundary → Type → Coreference |
| **RelationExtractor** | RHF pipeline + open-world | Discovers new relation types |
| **EvidenceLinker** | Links triples to evidence | Cross-document references |

### Coordinator Agents (Validation)

| Agent | Role | Key Features |
|-------|------|--------------|
| **ExtractionValidator** | Validates + iterative refinement | Up to 4 iterations, threshold ≥0.85 |
| **ExtractionVerificationAgent** | Final verification | Anti-hallucination, source grounding |
| **KnowledgeOrganizer** | KG integration | Entity dedup, relation normalization |

### Agent Base Class

All agents inherit from `BaseAgent` which provides:
- `call_llm()` / `call_llm_with_self_consistency()` - LLM interaction with confidence
- `store_in_memory()` / `retrieve_from_memory()` - SharedMemory access
- `post_hypothesis()` / `vote_on_hypothesis()` - Blackboard operations
- `send_message()` / `receive_messages()` - MessageBus communication
- `escalate_to_coordinator()` - Escalation for low-confidence items
- `request_refinement()` / `provide_feedback()` - Collaborative refinement

### Novel Features

**SharedMemory System:**
- Episodic memory: Document-specific context
- Semantic memory: Persistent facts and discovered relations
- Working memory: Current processing state
- Blackboard: Hypothesis posting and multi-agent voting

**MessageBus Communication:**
- Inter-agent messaging (INFORM, REQUEST, PROPOSE, DELEGATE, FEEDBACK)
- Escalation for low-confidence items
- Collaborative refinement requests

**Self-Consistency Confidence:**
- Multiple LLM samples with voting for confidence estimation
- Quality threshold (≥0.85) for acceptance
- Automatic escalation when confidence is low

**Open-World Extraction:**
- Discovery of new relation types not predefined
- Relation type learning across documents
- Cross-document entity resolution

## Installation

```bash
pip install -e .
```

## Usage

```python
from multi_agent_kg.core import DeliberativeOrchestrator, KnowledgeGraph, LLMConfig

# Configure
llm_config = LLMConfig(model="gpt-4o-mini", temperature=0.3)

# Create orchestrator with all features
orchestrator = DeliberativeOrchestrator(
    llm_config=llm_config,
    knowledge_graph=KnowledgeGraph(),
    quality_threshold=0.85,
    max_refinement_iterations=4,
    enable_self_consistency=True,
    enable_open_world=True,
    enable_cross_document=True,
)

# Process documents
result = orchestrator.process_document(text="Your document text here...")

# Process corpus
results = orchestrator.process_corpus([
    {"text": "Document 1...", "id": "doc1"},
    {"text": "Document 2...", "id": "doc2"},
])

# Export
export = orchestrator.export()

```

## Model Tiers

The system supports tiered model selection:
- **Small** (gpt-3.5-turbo): Simple tasks (document processing, domain classification)
- **Medium** (gpt-4o-mini): Core extraction (entities, relations, evidence)
- **Large** (gpt-4o): Coordination tasks (validation, verification)

## Requirements

- Python 3.11+
- OpenAI API key
- See requirements.txt for dependencies

## Project Structure

```
multi_agent_kg/
├── agents/
│   ├── base.py                    # Enhanced base agent with memory/comm
│   ├── document_processor.py      # Worker: Document ingestion
│   ├── domain_classifier.py       # Worker: Domain classification
│   ├── entity_extractor.py        # Worker: Multi-stage entity extraction
│   ├── relation_extractor.py      # Worker: RHF relation extraction
│   ├── evidence_linker.py         # Worker: Evidence linking
│   ├── extraction_validator.py    # Coordinator: Validation
│   ├── extraction_verification_agent.py  # Coordinator: Verification
│   └── knowledge_organizer.py     # Coordinator: KG integration
├── core/
│   ├── deliberative_orchestrator.py  # Main integrated orchestrator
│   ├── memory.py                  # SharedMemory + Blackboard
│   ├── communication.py           # MessageBus + CollaborationProtocol
│   └── knowledge_graph.py         # KG data structures
└── examples/
    └── deliberative_pipeline.py   # Demo script
```
