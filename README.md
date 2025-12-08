# Multi-Agent Knowledge Graph Enrichment Framework

A modular, general-purpose framework for building knowledge graphs using multiple LLM-backed agents. Inspired by KARMA but designed for flexibility across domains.

## Overview

This framework orchestrates multiple specialized agents to collaboratively:
- **Ingest** documents from various sources
- **Segment** and **summarize** content
- **Extract** entities and their types
- **Extract** relations using configurable schemas
- **Align** knowledge with existing ontologies
- **Detect** and **resolve** conflicts
- **Integrate** verified triples into a knowledge graph

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for each task
- 🔧 **Configurable Relation Schemas**: Define your own relation types
- 🌐 **Open-World Mode**: Allow discovery of new relation types
- ⚡ **Conflict Detection**: Automatic detection and resolution
- ✅ **Verification Layer**: LLM-based triple verification
- 🎯 **Domain-Agnostic**: Works across different knowledge domains

## Architecture

```
┌─────────────┐
│  Document   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Orchestrator                            │
│  ┌──────────────────────────────────────────────┐  │
│  │  Ingestion → Segmentation → Summarization    │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Entity Extraction → Relation Extraction     │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Schema Alignment → Conflict Detection       │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Verification → Knowledge Graph Integration  │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Knowledge  │
│    Graph    │
└─────────────┘
```

## Agents

1. **IngestionAgent**: Load documents from files or raw text
2. **SegmenterAgent**: Split documents into manageable chunks
3. **SummarizerAgent**: Generate concise summaries using LLM
4. **EntityAgent**: Extract named entities with types
5. **RelationAgent**: Extract relations based on configurable schemas
6. **SchemaAgent**: Align entities and relations with existing schema
7. **ConflictAgent**: Detect and flag conflicting triples
8. **VerifierAgent**: Score and approve triples based on evidence

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd multi-agent-kg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Configuration

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Quick Start

```python
from multi_agent_kg.core.orchestrator import Orchestrator
from multi_agent_kg.core.config import RelationType, RelationSchema

# Define relation schema
relation_schema = RelationSchema(
    types=[
        RelationType(
            name="treats",
            description="A treatment for a medical condition",
            allowed_subject_types=["Drug", "Therapy"],
            allowed_object_types=["Disease", "Condition"]
        ),
        RelationType(
            name="causes",
            description="A causal relationship between entities",
        ),
    ]
)

# Create orchestrator
orchestrator = Orchestrator(relation_schema=relation_schema)

# Process a document
sample_text = """
Aspirin is commonly used to treat headaches and reduce fever.
Studies show that smoking causes lung cancer and heart disease.
"""

kg = orchestrator.process_document(sample_text)

# Print results
kg.print_graph()
```

### Run Example Pipeline

```bash
python -m multi_agent_kg.examples.run_pipeline
```

## Project Structure

```
multi_agent_kg/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── knowledge_graph.py    # KG representation & operations
│   ├── messages.py            # Message types for agent communication
│   ├── orchestrator.py        # Multi-agent workflow orchestration
│   └── config.py              # Configuration classes
├── llm/
│   ├── __init__.py
│   └── openai_client.py       # OpenAI SDK wrapper
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── ingestion_agent.py     # Document ingestion
│   ├── segmenter_agent.py     # Text segmentation
│   ├── summarizer_agent.py    # Content summarization
│   ├── entity_agent.py        # Entity extraction
│   ├── relation_agent.py      # Relation extraction
│   ├── schema_agent.py        # Schema alignment
│   ├── conflict_agent.py      # Conflict detection
│   └── verifier_agent.py      # Triple verification
└── examples/
    ├── __init__.py
    └── run_pipeline.py        # Example end-to-end pipeline
```

## Advanced Features

### Custom Relation Schemas

Define domain-specific relation types:

```python
from multi_agent_kg.core.config import RelationType, RelationSchema

custom_schema = RelationSchema(
    types=[
        RelationType(
            name="employed_by",
            description="Employment relationship",
            allowed_subject_types=["Person"],
            allowed_object_types=["Organization"]
        ),
        RelationType(
            name="located_in",
            description="Geographic location",
            allowed_subject_types=["Organization", "Person"],
            allowed_object_types=["Location"]
        ),
    ]
)
```

### Open-World Relation Discovery

Enable the system to discover new relation types:

```python
# In RelationAgent
triples = relation_agent.run(
    text_chunks=segments,
    entities=entities,
    open_world=True  # Allow new relation types
)
```

### Conflict Resolution

The framework automatically detects conflicts (same subject and relation, different object):

```python
# Conflicts are logged and can be resolved programmatically
conflicts = conflict_agent.run(candidate_triples, kg)
for conflict in conflicts:
    print(f"Conflict: {conflict.subject} - {conflict.relation}")
    print(f"  Existing: {conflict.existing_object}")
    print(f"  New: {conflict.new_object}")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black multi_agent_kg/
```

### Type Checking

```bash
mypy multi_agent_kg/
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Inspired by the KARMA (Knowledge Graph Augmented Retrieval with Multi-Agent) framework, adapted for general-purpose knowledge graph construction.
