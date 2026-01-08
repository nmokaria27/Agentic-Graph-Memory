# Multi-Agent Knowledge Graph Framework

A deliberative multi-agent system for knowledge graph construction from unstructured text using LLM-backed agents with inter-agent communication, voting, and debate mechanisms.

## Overview

This framework implements an 8-agent pipeline where specialized AI agents collaborate through shared memory, message passing, and democratic deliberation to extract high-quality knowledge graphs from documents.

**Key Features:**
- **Multi-Agent Deliberation**: Agents vote and debate on uncertain extractions
- **Tiered LLM Usage**: Small/medium/large models for cost-efficiency
- **Anti-Hallucination**: Verification against source text with evidence linking
- **Domain-Adaptive**: Automatic schema generation for different document types
- **Open-World Relations**: Discovers novel relation types beyond predefined schemas

```
                              YOUR DOCUMENT
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SHARED MEMORY                                       │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────────────┐   │
│   │   Memory    │   │  Blackboard │   │         Message Bus             │   │
│   │  (context)  │   │  (debates)  │   │    (agent communication)        │   │
│   └─────────────┘   └─────────────┘   └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
        ═══════════════════════════╪═══════════════════════════════════
                    EXTRACTION     │
        ═══════════════════════════╪═══════════════════════════════════
                                   ▼
              ┌────────────────────────────────────────┐
              │  [1] Document Processor                │
              │      Chunks text into segments         │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [2] Domain Classifier                 │
              │      Scientific? Legal? News?          │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [3] Entity Extractor                  │
              │      4-stage: find → refine → type     │
              │              → resolve duplicates      │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [4] Relation Extractor                │
## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       SHARED MEMORY LAYER                        │
│  • Episodic/Semantic/Procedural Memory                          │
│  • Blackboard (voting/debate)                                   │
│  • MessageBus (agent communication)                              │
└──────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────  PIPELINE  ─────────────────────────────┐
│                                                                   │
│  [1] DocumentProcessor  →  Segment text into chunks              │
│  [2] DomainClassifier   →  Identify domain + generate schema    │
│  [3] EntityExtractor    →  4-stage entity extraction             │
│  [4] RelationExtractor  →  RHF pipeline + open-world discovery  │
│  [5] EvidenceLinker     →  Link facts to source text            │
│                                                                   │
│  ┌─────────────── DELIBERATION PHASE ───────────────┐            │
│  │  [6] Multi-Agent Voting & Debate                 │            │
│  │      • Low confidence → Post to blackboard       │            │
│  │      • 2-3 agents vote (weighted by role)        │            │
│  │      • Conflicts trigger debate loops            │            │
│  └──────────────────────────────────────────────────┘            │
│                                                                   │
│  [7] ExtractionValidator       →  Iterative refinement          │
│  [8] VerificationAgent         →  Anti-hallucination check      │
│  [9] KnowledgeOrganizer        →  Dedup + normalize + store     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                              ▼
                      KNOWLEDGE GRAPH
```

## Agent API Reference

### Agent 1: DocumentProcessor
- **Role**: Text segmentation and preprocessing  
- **Model**: gpt-3.5-turbo (Small)
- **Input**: Raw document text
- **Output**: List of text segments (1500-2000 chars each)
- **LLM Calls**: 0 (rule-based)

### Agent 2: DomainClassifier
- **Role**: Domain identification and schema generation  
- **Model**: gpt-4o-mini (Medium)
- **Input**: Document segments
- **Output**: Domain + entity_types + relation_types + examples
- **LLM Calls**: 2 (domain analysis, relation examples)
- **Side Effects**: Broadcasts schema to all downstream agents

### Agent 3: EntityExtractor
- **Role**: 4-stage entity extraction pipeline  
- **Model**: gpt-4o-mini (Medium)
- **Stages**:
  1. Initial extraction (with domain entity types)
  2. Boundary refinement (fix text spans)
  3. Type assignment (validate types)
  4. Coreference resolution (merge duplicates)
- **Input**: Text segments + entity_types
- **Output**: Entities with types, confidence, aliases
- **LLM Calls**: 4 per segment
- **Side Effects**: Posts low-confidence entities for voting

### Agent 4: RelationExtractor
- **Role**: Relation-Head-First (RHF) triple extraction  
- **Model**: gpt-4o-mini (Medium)
- **Stages**:
  1. Relation identification
  2. Head entity binding (subjects)
  3. Tail entity binding (objects)
- **Input**: Entities + relation_types
- **Output**: Triples (subject, relation, object) with evidence
- **LLM Calls**: 3 per segment
- **Side Effects**: Discovers new relation types if open-world enabled

### Agent 5: EvidenceLinker
- **Role**: Link triples to source evidence  
- **Model**: gpt-4o-mini (Medium)
- **Input**: Triples from RelationExtractor
- **Output**: Triples with evidence sentences + confidence scores
- **LLM Calls**: ~N/10 batches
- **Side Effects**: Adjusts confidence based on evidence quality

### Agent 6: Multi-Agent Deliberation
- **Coordinator**: DeliberationCoordinator  
- **Voting Panel**: EntityExtractor, RelationExtractor, EvidenceLinker
- **Process**:
  1. Low-confidence items posted to blackboard
  2. Voting agents cast weighted votes (7 levels: STRONG_ACCEPT to STRONG_REJECT)
  3. Consensus threshold: 0.6 (60% weighted agreement)
  4. Conflicts → debate phase with arguments
- **Output**: Accepted/rejected items with rationales

### Agent 7: ExtractionValidator
- **Role**: Validation coordinator with iterative refinement  
- **Model**: gpt-4o (Large)
- **Input**: Low-confidence entities/triples from workers
- **Output**: Validated extractions with quality scores
- **LLM Calls**: 2-4 per iteration
- **Max Iterations**: 4 (stops when quality ≥ threshold)

### Agent 8: ExtractionVerificationAgent
- **Role**: Anti-hallucination verification  
- **Model**: gpt-4o (Large)
- **Process**:
  1. Source verification (check against text)
  2. Cross-document consistency check
  3. Classify: verified/partial/rejected/hallucinated
- **Input**: Validated triples
- **Output**: Approved/rejected with verification_status
- **LLM Calls**: ~2 operations (batch verification)

### Agent 9: KnowledgeOrganizer
- **Role**: KG integration and maintenance  
- **Model**: gpt-4o-mini (Medium)
- **Process**:
  1. Entity deduplication (merge aliases)
  2. Relation normalization
  3. KG integration
  4. Export to JSON
- **Input**: Verified entities/triples
- **Output**: Updated KnowledgeGraph + statistics
- **LLM Calls**: ~2 operations (dedup, normalization)

## Installation

```bash
git clone https://github.com/PranavBykampadi/multi-agent-kg.git
cd multi-agent-kg
pip install -e .
```

## Quick Start

```python
from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig

documents = [{"id": "doc1", "text": "Your document text...", "metadata": {}}]

orchestrator = DeliberativeOrchestrator(
    llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.2),
    quality_threshold=0.7,
    enable_deliberation=True
)

results = orchestrator.process_corpus(documents)
kg = results["knowledge_graph"]
kg.export("output.json")
```

## Configuration

**Model Tiers:**
- Small (gpt-3.5-turbo): Simple tasks (segmentation)
- Medium (gpt-4o-mini): Extraction tasks (entities, relations)
- Large (gpt-4o): Validation and verification

**Key Parameters:**
- `quality_threshold`: Minimum confidence for acceptance (default: 0.7)
- `max_refinement_iterations`: Iterative refinement limit (default: 4)
- `enable_deliberation`: Multi-agent voting/debate (default: True)
- `enable_open_world`: Discover new relation types (default: True)
- `consensus_threshold`: Voting agreement needed (default: 0.6)

## Data Structures

**Entity:**
```python
{
    "id": "entity_001",
    "text": "intimate partner violence",
    "type": "Violence_Type",
    "confidence": 0.92,
    "start": 45,
    "end": 70,
    "aliases": ["IPV"]
}
```

**Triple:**
```python
{
    "subject": "intimate partner violence",
    "subject_id": "entity_001",
    "relation": "associated_with",
    "object": "depression",
    "object_id": "entity_023",
    "confidence": 0.85,
    "evidence": {
        "text": "IPV was significantly associated with depression...",
        "type": "explicit",
        "char_start": 1234
    }
}
```

