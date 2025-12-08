# Multi-Agent Knowledge Graph Framework

Building knowledge graphs from unstructured text is tricky—entities hide in weird places, relationships aren't always obvious, and LLMs love to hallucinate facts that sound right but aren't. This framework uses a team of 8 specialized AI agents that each handle one part of the problem, and they actually communicate with each other through a shared memory system. When an agent isn't confident about something, it doesn't just guess—it posts the question to a blackboard where other agents vote and debate until they reach consensus.

The pipeline has 9 steps: chunk the document, classify its domain, extract entities (4-stage process), find relationships, link everything to evidence, run multi-agent deliberation on uncertain items, validate quality, do an anti-hallucination check, and organize it into a knowledge graph. The deliberation step is where it gets interesting—agents cast weighted votes (coordinator agents count more than workers), and if votes conflict, they enter a debate loop where they present arguments until one side wins. Everything flows through shared memory so context carries across the whole system.

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
              │      RHF pipeline + discovers new      │
              │      relation types not in schema      │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [5] Evidence Linker                   │
              │      Links each fact to source text    │
              └──────────────────┬─────────────────────┘
                                 │
        ═══════════════════════════╪═══════════════════════════════════
                  DELIBERATION     │
        ═══════════════════════════╪═══════════════════════════════════
                                 ▼
              ┌────────────────────────────────────────┐
              │  [6] Multi-Agent Deliberation          │
              │                                        │
              │   Low confidence? → Post to blackboard │
              │                          ↓             │
              │              Other agents vote         │
              │                          ↓             │
              │         Votes conflict? → Debate loop  │
              │                          ↓             │
              │              Accept or Reject          │
              └──────────────────┬─────────────────────┘
                                 │
        ═══════════════════════════╪═══════════════════════════════════
                  VALIDATION       │
        ═══════════════════════════╪═══════════════════════════════════
                                 ▼
              ┌────────────────────────────────────────┐
              │  [7] Extraction Validator              │
              │      Iterates until quality ≥ 85%      │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [8] Verification Agent                │
              │      Anti-hallucination check          │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  [9] Knowledge Organizer               │
              │      Dedupes, normalizes, stores       │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                        KNOWLEDGE GRAPH
```

## Key Features

- **Multi-agent voting & debate** - Agents vote on uncertain extractions with 7-level weighted votes. Conflicts trigger debate loops with arguments until consensus.
- **Self-consistency confidence** - Multiple LLM samples vote on answers. If 4/5 agree, confidence is 0.8. Disagreement = escalation.
- **Open-world relation discovery** - Finds relation types not predefined in the schema.
- **Evidence grounding** - Every fact links to source text. Can't prove it? Probably hallucinated.
- **Cross-document memory** - Context persists across documents for entity resolution.

## Quick Start

```bash
pip install -e .
export OPENAI_API_KEY=your_key
python -m multi_agent_kg.examples.deliberative_pipeline
```

```python
from multi_agent_kg.core import DeliberativeOrchestrator, KnowledgeGraph, LLMConfig

orchestrator = DeliberativeOrchestrator(
    llm_config=LLMConfig(model="gpt-4o-mini"),
    knowledge_graph=KnowledgeGraph(),
)

result = orchestrator.process_document(text="Your text here...")
```
