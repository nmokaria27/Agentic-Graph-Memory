# Agent-Graph-Memory vs. Cognee — architecture comparison

Both systems build a knowledge graph (KG) from raw text and answer questions with it, but they differ in the core abstraction, control philosophy, storage, and QA style.

---

## 1. Agent-Graph-Memory (your system)

### Central abstraction
A **Governed Knowledge Graph** `G = (E, T, D, φ, γ)` (see `README.md` and `multi_agent_kg/core/governed_kg.py`).
- `E` entities, `T` triples, `D` governed domains.
- `φ: E → 2^D` assigns entity ownership to domains.
- `γ` routes every proposed triple to a domain expert for a decision: `approve | reject | revise | escalate | auto_approve`.
- Governance modes: `strict`, `permissive`, `audit_only`, `triage`.
- Every decision is written to an audit log (`_audit_log` in `GovernedKnowledgeGraph`).

### Construction pipeline
A **multi-agent extraction pipeline** with 9 sequential stages (`AGENT_SCHEMA.md` §2.1; `multi_agent_kg/core/deliberative_orchestrator.py`):
1. `DocumentProcessor` — segments text.
2. `DomainClassifier` — infers domain + schema.
3. `EntityExtractor` — 4-stage entity extraction (initial → boundary → type → coreference) with self-consistency.
4. `RelationExtractor` — 3-stage RHF (relation-head-first) triple extraction, open-world relation discovery.
5. `EvidenceLinker` — attaches sentence-level evidence and cross-references prior triples.
6. `DeliberationCoordinator` — multi-agent voting/debate on uncertain items (confidence 0.35–0.65).
7. `ExtractionValidator` — iterative refinement.
8. `ExtractionVerificationAgent` — final anti-hallucination check.
9. `KnowledgeOrganizer` — deduplication, normalization, integration into the KG.

Shared infrastructure: `SharedMemory` (episodic, semantic, working, procedural + blackboard), `MessageBus` for inter-agent messages (`multi_agent_kg/core/memory.py`, `multi_agent_kg/core/communication.py`).

### Retrieval & QA
- Retrieval layer: `VectorIndex` (exact brute-force cosine search over `numpy` matrices) + `KGVectorStore` (entity, triple, domain indices) + graph traversal (`find_paths`, `neighbourhood` in `multi_agent_kg/core/graph_traversal.py`).
- QA layer: `QAOrchestrator` routes questions to `DomainExpertAgent` instances that answer from their owned subgraphs.
- Advanced QA: `AdvancedQAOrchestrator` with active exploration, multi-expert debate, critic agent, provenance tracking (`scripts/run_demo.py`).

### LLM backend
`multi_agent_kg/llm/openai_client.py` supports Ollama (default), OpenAI, and vLLM with model tiers (`SMALL`, `MEDIUM`, `LARGE`).

### Evaluation & tuning
- **KGAFE** (KG-Grounded Atomic Fact Evaluation): decomposes answers into atomic facts, verifies them in 3 tiers (exact match → path-based → semantic), then uses a judge panel (`evaluation/kgafe/`).
- **SciERC benchmark** for extraction quality.
- **Governance benchmark** for routing and approval decisions.
- **Tuning**: two-stage Optuna TPE (`scripts/tune/tune_retrieval.py`, `scripts/tune/tune_chunk.py`): Stage A tunes retrieval/answer-format parameters on frozen KGs; Stage B sweeps `chunk_size` and re-tunes retrieval.

### Storage
KG is in-memory dataclasses (`KnowledgeGraph`, `Entity`, `Triple` in `multi_agent_kg/core/knowledge_graph.py`); serialized to JSON. Vector indices are `npz` files. No external graph or vector database.

---

## 2. Cognee

### Central abstraction
An **AI memory platform** with an **Extract–Cognify–Load (ECL)** pipeline (`cognee.tex` and `cognee_appendix.tex` in the paper; Cognee `CLAUDE.md`).
- **Extract**: ingest heterogeneous inputs (text, PDFs, images, audio, code, URLs).
- **Cognify**: schema-based transformation via Pydantic models → entities, relations, attributes, summaries.
- **Load**: write to graph, relational, and vector stores.

Public API: `add()`, `cognify()`, `search()`, `memify()` (or `remember/recall/forget/improve`).

### Construction pipeline
1. **Ingestion** — load files/URLs/text, record metadata in a relational DB.
2. **Tagging** — classify by MIME type, merge metadata, deduplicate by content hash, organize into datasets.
3. **Chunking** — token-limited chunks.
4. **Graph construction** — LLM extracts entities/relations into structured schema objects (uses Instructor), produces graph fragments.
5. **Indexing** — write to graph DB, relational DB, and vector DB.

Pipeline is **task-based** (`cognee/modules/pipelines/`), not agent-based. Tasks are composable and can run sequentially or in parallel.

### Storage adapters
Cognee uses adapter interfaces for multiple backends:
- **Graph DB**: Kuzu (default), Neo4j, Neptune, Postgres via `GraphDBInterface`.
- **Vector DB**: LanceDB (default), ChromaDB, PGVector, Qdrant, Weaviate, Milvus via `VectorDBInterface`.
- **Relational DB**: SQLite (default), PostgreSQL.

### Retrieval & QA
Many search strategies in `cognee/modules/search/types/SearchType.py`:
- `GRAPH_COMPLETION` (default) — graph traversal + LLM completion.
- `GRAPH_SUMMARY_COMPLETION` — pre-computed summaries + graph context.
- `GRAPH_COMPLETION_COT` — chain-of-thought over graph.
- `TRIPLET_COMPLETION` — triplet-based search.
- `RAG_COMPLETION` — traditional RAG with chunks.
- `CHUNKS` / `CHUNKS_LEXICAL` — vector/keyword search over chunks.
- `SUMMARIES` — search document summaries.
- `CYPHER` — direct Cypher query.
- `NATURAL_LANGUAGE` — natural language → structured query.
- `TEMPORAL` — time-aware graph search.
- `FEELING_LUCKY` — automatic strategy selection.

QA is generally a retriever + LLM completion; no domain-expert agent layer or debate arena.

### LLM backend
`LLMGateway.py` supports OpenAI, Anthropic, Gemini, Ollama, Mistral, Bedrock, and others. Uses Instructor for structured output. Default model is `openai/gpt-4o-mini`.

### Evaluation & tuning
- **Evaluation**: multi-hop QA benchmarks (HotPotQA, TwoWikiMultiHop, MuSiQue) scored with exact match, token-level F1, and DeepEval LLM-based correctness.
- **Tuning**: **Dreamify** framework. Treats the full pipeline as an objective function and optimizes 6 parameters with TPE:
  - `chunk_size` (200–2000 tokens)
  - `search_type` (e.g., `cognee_completion` vs `cognee_graph_completion`)
  - `top_k`
  - `qa_system_prompt`
  - `graph_prompt`
  - `task_getter_type` (summary generation on/off)

### Multi-tenancy & deployment
- User → Dataset → Data hierarchy with permission filtering (`ENABLE_BACKEND_ACCESS_CONTROL=True`).
- Docker images, CLI (`cognee-cli`), MCP server, UI, and API layer.

---

## 3. Head-to-head differences

| Dimension | Agent-Graph-Memory (your system) | Cognee |
|-----------|----------------------------------|--------|
| **Core abstraction** | Governed KG with domain ownership and explicit governance decisions | General-purpose AI memory / KG platform with ECL pipeline |
| **Governance** | Yes: `φ` ownership, `γ` routing, audit log, modes (strict/permissive/triage/audit_only) | No equivalent domain-ownership/governance layer |
| **Pipeline style** | Multi-agent pipeline (9 agents, deliberation, message bus, shared memory) | Task-based pipeline (ingest → chunk → extract → index) |
| **Extraction** | Multi-stage, multi-agent, deliberation, evidence linking, cross-document, open-world relations | Single LLM-based extraction task using Instructor/Pydantic schema |
| **Storage** | In-memory KG + JSON + custom numpy vector index | Pluggable graph DB + vector DB + relational DB |
| **Retrieval** | Domain-routed subgraphs, graph traversal, exact cosine vector search | Suite of search strategies (RAG, chunks, summaries, graph completion, Cypher, temporal, etc.) |
| **QA** | Domain expert agents + advanced QA with debate, critic, provenance | Retriever + LLM completion (various search types) |
| **Memory model** | Episodic/semantic/working/procedural + blackboard | Vector store + graph DB + session memory (in `remember`) |
| **Tuning** | Two-stage Optuna: retrieval params on frozen KGs, then chunk-size grid | End-to-end Dreamify TPE over chunk, retrieval, prompts, task getter |
| **Evaluation** | KGAFE (KG-grounded answer faithfulness), SciERC extraction, governance benchmarks | HotPotQA / 2WikiMultiHop / MuSiQue with EM/F1/DeepEval correctness |
| **Multi-tenancy** | Not present | Built-in user/dataset access control |
| **Deployment** | Python scripts + React frontend | Python package, CLI, Docker, MCP server, UI, API |

---

## 4. Similarities

Both:
- Are modular LLM-based KG construction + retrieval systems.
- Use chunking before LLM extraction.
- Combine vector embeddings and graph structure for retrieval.
- Target multi-hop question answering.
- Use hyperparameter optimization (Optuna/TPE) over chunk size and retrieval parameters.
- Support local LLMs (Ollama) and cloud LLMs (OpenAI).

---

## 5. When each design wins

- **Your system** is better when the priority is:
  - Auditable, governed updates (domain ownership, approve/reject, audit log).
  - Multi-agent deliberation and evidence-grounded extraction.
  - QA that explicitly routes to domain experts and can debate/criticize.
  - Research on extraction faithfulness and governance.

- **Cognee** is better when the priority is:
  - Production-grade deployment with persistent, scalable graph/vector/relational stores.
  - Broad retrieval strategy menu (Cypher, temporal, summaries, RAG, etc.).
  - Multi-tenant agent memory, Docker/CLI/MCP integration.
  - End-to-end hyperparameter optimization over the whole pipeline and prompt set.

---

*Generated from the paper “Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning” (arXiv:2505.24478) and the current Agent-Graph-Memory codebase (`feat/vector-index-and-qa-improvements`).*
