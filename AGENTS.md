# AGENTS.md — Multi-Agent Knowledge Graph Framework

This document is the single source of truth for AI agents and developers working on this project. It describes architecture, agents, data structures, entry points, and conventions.

---

## 1. Project Overview

**Name:** Multi-Agent Knowledge Graph Framework (`multi-agent-kg`)

**Purpose:** Build high-quality knowledge graphs from unstructured text using a **deliberative multi-agent system**. Specialized LLM-backed agents collaborate via shared memory, message passing, voting, and debate to extract entities and relations, verify against source text, and integrate into a unified KG.

**Key characteristics:**
- **8-agent pipeline:** DocumentProcessor → DomainClassifier → EntityExtractor → RelationExtractor → EvidenceLinker → (Deliberation) → ExtractionValidator → ExtractionVerificationAgent → KnowledgeOrganizer
- **Tiered LLM usage:** Small/medium/large model tiers for cost vs quality (configurable; default uses Ollama `gemma3:27b` for all tiers)
- **Anti-hallucination:** Verification agent checks extractions against source text; evidence linking ties triples to spans
- **Domain-adaptive:** Domain classifier infers domain and schema from content; no fixed schema required
- **Open-world relations:** Relation extractor can discover new relation types beyond any predefined schema
- **Multi-agent deliberation:** Low-confidence extractions go to a blackboard; other agents vote; conflicts trigger debate; consensus determines accept/reject

**Repository layout (high level):**
- `multi_agent_kg/` — main package
  - `agents/` — pipeline agents (document_processor, domain_classifier, entity_extractor, relation_extractor, evidence_linker, extraction_validator, extraction_verification_agent, knowledge_organizer)
  - `core/` — orchestrator, memory, communication, deliberation, knowledge graph, config, kg_operations, incremental_enrichment, domain_experts
  - `llm/` — LLM client (Ollama + OpenAI)
  - `utils/` — debug logger, KG visualizer
  - `legacy_agents/` — older pipeline variants (reference only)
- Root scripts: `run_pipeline_on_text.py`, `run_incremental_enrichment.py`, `run_domain_qa.py`, `extract_pdf.py`

---

## 2. Architecture

### 2.1 High-Level Flow

```
                    SHARED MEMORY LAYER
  • Episodic / Semantic / Procedural memory
  • Blackboard (hypotheses, votes)
  • MessageBus (agent messages)
                              │
                              ▼
  ┌────────────────────────── PIPELINE ──────────────────────────────┐
  │  [1] DocumentProcessor    → segment text into chunks              │
  │  [2] DomainClassifier     → domain + schema (entity/relation)     │
  │  [3] EntityExtractor      → 4-stage entity extraction             │
  │  [4] RelationExtractor    → RHF triple extraction + open-world    │
  │  [5] EvidenceLinker       → link triples to source evidence       │
  │  ┌──────────── DELIBERATION ────────────┐                        │
  │  │ [6] Low confidence → blackboard      │                        │
  │  │     Voting (Entity/Relation/Evidence)│                        │
  │  │     Conflict → debate → resolution  │                        │
  │  └──────────────────────────────────────┘                        │
  │  [7] ExtractionValidator  → iterative refinement                 │
  │  [8] ExtractionVerificationAgent → anti-hallucination check       │
  │  [9] KnowledgeOrganizer   → dedup, normalize, KG integration     │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      KNOWLEDGE GRAPH
```

### 2.2 Shared Memory (`core/memory.py`)

- **MemoryType:** `EPISODIC` (document context), `WORKING` (current task), `SEMANTIC` (extracted knowledge), `PROCEDURAL` (patterns).
- **Store/retrieve:** `store(memory_type, content, source, references, metadata)` → entry id; `retrieve(memory_type, source, entity, limit)`.
- **Blackboard:** `post_to_blackboard(author, entry_type, content)`, `vote_on_blackboard(entry_id, voter, confidence)`, `respond_to_blackboard`, `resolve_blackboard_entry`. Entry types: `hypothesis`, `request`, `refinement`, `conflict`, `vote`.
- **Cross-document:** `entity_aliases` (alias → canonical_id), `entity_contexts`, `register_entity_alias`, `add_entity_context`, `get_entity_context`.
- **Document tracking:** `register_document`, `processed_documents`, `get_document_history`.

### 2.3 Message Bus (`core/communication.py`)

- **MessageBus:** `send(sender, receiver, comm_type, content, ...)`, `receive(agent_name, comm_type, priority)`, `reply(original_msg_id, sender, comm_type, content)`, inboxes per agent, optional topic subscribe.
- **CommunicationType:** INFORM, REQUEST, RESPONSE, PROPOSE, ACCEPT, REJECT, REFINE, DELEGATE, FEEDBACK.
- **MessagePriority:** LOW, NORMAL, HIGH, URGENT.
- **CollaborationProtocol:** `request_refinement`, `propose_hypothesis`, `vote_on_proposal`, `delegate_task`, `provide_feedback`.

### 2.4 Deliberation (`core/deliberation.py`)

- **DeliberationCoordinator:** Receives hypotheses from workers; requests votes from voting agents; detects conflict (score near zero); runs debate; resolves with weighted consensus.
- **VoteType:** STRONG_ACCEPT, ACCEPT, WEAK_ACCEPT, ABSTAIN, WEAK_REJECT, REJECT, STRONG_REJECT (with weights ±1.0, ±0.75, ±0.5, 0).
- **Hypothesis:** id, author, hypothesis_type (entity/triple/relation), content, initial_confidence, evidence, status (PENDING, VOTING, DEBATING, ACCEPTED, REJECTED, …), votes, debate_arguments.
- **Thresholds:** consensus_threshold (default 0.6), min_votes (default 2), CONFLICT_THRESHOLD 0.3. Agent weights: EvidenceLinker 1.2, ExtractionValidator/VerificationAgent 1.5, DomainClassifier 0.8, DocumentProcessor 0.5.
- **Orchestrator integration:** Only items in the “uncertain” band (e.g. confidence 0.35–0.65) are sent to deliberation; above 0.65 accept, below 0.35 reject without vote. Orchestrator simulates votes from DomainClassifier, EvidenceLinker, RelationExtractor (for entities) and EntityExtractor, EvidenceLinker, ExtractionValidator (for triples), then runs debate/resolution as needed.

---

## 3. Pipeline: Step-by-Step

Execution is driven by `DeliberativeOrchestrator.process_document()` (and `process_corpus()` for multiple docs).

| Step | Agent | Input | Output | Notes |
|------|--------|--------|--------|--------|
| 1 | DocumentProcessor | context.text, optional source_path | Segments (list of chunks 1500–2000 chars, overlap 150) | Rule-based segmentation; stores doc in shared memory |
| 2 | DomainClassifier | segments | domain + entity_types + relation_types + examples | 2 LLM calls (domain analysis, relation examples); result in context.domain and domain_config |
| 3 | EntityExtractor | segments, domain_config | Entities (name, type, spans, confidence, aliases) | 4-stage: initial extraction → boundary refinement → type assignment → coreference; self-consistency optional |
| 4 | RelationExtractor | segments, entities, domain_config | Triples (subject, predicate, object, confidence, evidence) | RHF: relation identification → head binding → tail binding; open-world can add new relation types |
| 5 | EvidenceLinker | triples, segments | Triples with evidence sentences and updated confidence | Batched LLM calls |
| 6 | Deliberation | entities, triples (low confidence only), segments | Refined entity/triple lists after voting/debate | Only 0.35 ≤ confidence < 0.65; voting agents + debate; accepted/rejected counts in results |
| 7 | ExtractionValidator | entities, triples | Validated entities/triples, refinement iterations | Iterative refinement up to max_refinement_iterations (default 4); quality threshold |
| 8 | ExtractionVerificationAgent | validated entities/triples | approved_triples, rejected_triples, verification_status | Source verification + cross-doc consistency; classifies verified/partial/rejected/hallucinated |
| 9 | KnowledgeOrganizer | verified entities, approved_triples | KG updated; kg_stats (total_entities, total_triples, etc.) | Dedup, relation normalization, add_entity/add_triple |

---

## 4. Agent Reference

### 4.1 Base Agent (`agents/base.py`)

- **BaseAgent:** Abstract base with name, role (WORKER/COORDINATOR), knowledge_graph, shared_memory, message_bus, llm_config, model_tiers, quality_threshold, max_iterations.
- **ModelTier:** SMALL, MEDIUM, LARGE (maps to model names in orchestrator/model_tiers).
- **ExtractionResult:** items (list), confidence, evidence, metadata, needs_escalation, escalation_reason.
- **AgentContext:** document_id, text, entities, relations, domain, iteration, max_iterations, quality_threshold, previous_feedback.
- **LLM:** `call_llm(prompt, system_prompt, tier, temperature, max_tokens, response_format)`; `call_llm_with_self_consistency(..., n_samples, temperature)` for confidence via majority vote.
- **Memory:** `store_in_memory`, `retrieve_from_memory`, `get_entity_context`.
- **Blackboard:** `post_hypothesis`, `vote_on_hypothesis`, `get_pending_hypotheses`, `resolve_hypothesis`.
- **Communication:** `send_message`, `receive_messages`, `request_refinement`, `delegate_task`, `provide_feedback`.
- **Escalation:** `escalate_to_coordinator(reason, items, context)` → blackboard + message to ExtractionValidator.
- **Deliberation:** `submit_for_deliberation`, `cast_vote`, `provide_debate_argument`, `check_for_vote_requests`, `check_for_debate_requests`, `evaluate_hypothesis_for_vote` (override in subclasses).

### 4.2 DocumentProcessor

- **Role:** Worker. Segments document into overlapping chunks.
- **Config:** min_segment_length=1500, max_segment_length=2000, overlap=150.
- **LLM:** None (rule-based).
- **Output:** ExtractionResult with items = list of segment dicts (text, start, end, metadata).

### 4.3 DomainClassifier

- **Role:** Worker. Infers domain and generates entity/relation schema from content.
- **Input:** segments.
- **Output:** Single domain config (domain, entity_types, relation_types, examples).
- **LLM:** ~2 calls (domain analysis, relation examples).
- **Side effect:** Result stored in context.domain and passed as domain_config to downstream agents.

### 4.4 EntityExtractor

- **Role:** Worker. Multi-stage entity extraction.
- **Stages:** (1) Initial extraction with domain entity types, (2) Boundary refinement, (3) Type assignment, (4) Coreference resolution.
- **Input:** segments, domain_config.
- **Output:** List of entities: name, type, start/end, confidence, aliases, evidence/source_spans.
- **LLM:** 4 calls per segment (or batched). Optional self-consistency for confidence.
- **Side effect:** Low-confidence entities can be submitted for deliberation.

### 4.5 RelationExtractor

- **Role:** Worker. RHF (Relation-Head-First) extraction.
- **Stages:** (1) Relation identification, (2) Head entity binding, (3) Tail entity binding.
- **Input:** segments, entities, domain_config.
- **Output:** Triples with subject/object (with entity refs), predicate, confidence, evidence.
- **LLM:** 3 calls per segment. Open-world can discover new relation types (stored in agent state; `get_discovered_relations()`).
- **Side effect:** New relation types broadcast for downstream use.

### 4.6 EvidenceLinker

- **Role:** Worker. Links triples to source text and adjusts confidence.
- **Input:** triples, segments.
- **Output:** Triples with evidence (text, type, char_start) and updated confidence.
- **LLM:** Batched (~N/10).
- **Config:** enable_cross_reference for cross-document evidence.

### 4.7 ExtractionValidator

- **Role:** Coordinator. Validates and refines extractions; runs iterative refinement.
- **Input:** entities, triples (after deliberation).
- **Output:** Validated dict with entities and triples; metadata includes refinement_iterations.
- **LLM:** 2–4 per iteration; tier LARGE. Stops when quality ≥ threshold or max_iterations.

### 4.8 ExtractionVerificationAgent

- **Role:** Coordinator. Anti-hallucination: verify triples against source text; cross-document consistency.
- **Input:** validated entities and triples.
- **Output:** approved_triples, rejected_triples, verification_status.
- **LLM:** ~2 batch operations (tier LARGE).

### 4.9 KnowledgeOrganizer

- **Role:** Coordinator. Final KG integration.
- **Process:** Entity deduplication (LLM merge groups), relation normalization (LLM), add_entity/add_triple to KnowledgeGraph.
- **Input:** verified entities, approved_triples only.
- **Output:** ExtractionResult with metadata kg_stats (total_entities, total_triples, unique_relations, etc.); also `export_knowledge_graph()`, `get_kg_stats()`.

---

## 5. Core Data Structures

### 5.1 Knowledge Graph (`core/knowledge_graph.py`)

- **Entity:** id, labels (list), type (optional), metadata.
- **Triple:** subject, relation, object, confidence, source, metadata.
- **Conflict:** same subject+relation, different object (existing_triple vs new_triple).
- **KnowledgeGraph:** `entities: Dict[str, Entity]`, `triples: List[Triple]`, `_triple_set` for dedup. Methods: `add_entity`, `add_triple`, `get_orphan_triples`, `find_conflicts`, `get_triples_by_subject/relation/object`, `to_dict`, `to_json`, `get_stats`. Does not auto-create entities for triples; KnowledgeOrganizer must ensure entity existence.

### 5.2 Config (`core/config.py`)

- **RelationType:** name, description, allowed_subject_types, allowed_object_types; `validate_triple(subject_type, object_type)`.
- **RelationSchema:** types (list of RelationType), allow_new_types.
- **LLMConfig:** model, temperature, max_tokens, top_p; `to_dict()`.

### 5.3 Adaptive Config (`core/adaptive_config.py`)

- **ModelSpec:** name, max_context_tokens, max_output_tokens, avg_chars_per_token.
- **MODEL_SPECS:** qwen3:4b/8b, gemma3:27b, mistral, deepseek-r1, gpt-3.5/4o/4o-mini, etc.
- **DomainSchema:** name, description, entity_types, relation_types, quality_threshold, extraction_examples, parent_domain, keywords.
- **DomainTaxonomy:** register_domain, get_domain; no built-in domains (agent-driven discovery).
- **AdaptiveBatchCalculator, ThresholdAutoTuner:** for context-aware batching and threshold tuning.

---

## 6. LLM and Environment

### 6.1 Backend (`llm/openai_client.py`)

- **Backend:** `LLM_BACKEND` env: `ollama` (default) or `openai`. OpenAI requires `OPENAI_API_KEY`.
- **Ollama:** `OLLAMA_BASE_URL` default `http://localhost:11434/v1` (e.g. SSH tunnel to GPU host).
- **Functions:** `chat_completion(messages, model, temperature, max_tokens)` → raw string; `chat_completion_json(...)` → parsed JSON (strips `<think>` blocks, handles markdown code fences and broken JSON).
- **Model mapping:** OpenAI names (gpt-4o, gpt-4o-mini, …) mapped to Ollama models when backend is Ollama. Quota fallback for OpenAI: gpt-4o → gpt-4o-mini → gpt-3.5-turbo.
- **Embeddings:** `get_embedding(text, model)`; on Ollama failure, returns deterministic hash-based placeholder vector.

### 6.2 Orchestrator Model Tiers

Default in `DeliberativeOrchestrator`: all tiers use `gemma3:27b`. Can override with `model_tiers: Dict[ModelTier, str]` (e.g. SMALL=qwen3:4b, MEDIUM=qwen3:8b, LARGE=gemma3:27b).

---

## 7. Entry Points and Scripts

### 7.1 Run Full Pipeline on Text

- **Script:** `run_pipeline_on_text.py`
- **Expects:** `gfy083_full_plaintext.txt` (or set `text_file`). Can be produced by `extract_pdf.py`.
- **Flow:** Load text → build document dict with id/metadata → `DeliberativeOrchestrator(llm_config, quality_threshold=0.6, max_refinement_iterations=1, enable_self_consistency=False, enable_open_world=True, enable_cross_document=False, debug_logger)` → `process_corpus(documents)` → export to `kg_export.json`, print stats, run KG visualizer (interactive HTML + static PNG).
- **Env:** `LLM_BACKEND`, `OPENAI_API_KEY` if OpenAI; `OLLAMA_BASE_URL` for Ollama.

### 7.2 Incremental Enrichment

- **Script:** `run_incremental_enrichment.py`
- **Module:** `core/incremental_enrichment.py`
- **Purpose:** Add new documents to an existing KG: run same extraction pipeline on new docs, compute diff (`core/kg_operations.compute_diff`), resolve conflicts with **ConflictResolver** (LLM), merge with `merge_kg`, return report.
- **ConflictResolver:** For each (existing_triple, candidate_triple) conflict, returns keep_existing / keep_new / keep_both / merge (with optional merged_triple).
- **IncrementalEnricher:** Wraps orchestrator, diff, resolve, merge and produces structured change report.

### 7.3 Domain Expert QA

- **Script:** `run_domain_qa.py`
- **Module:** `core/domain_experts.py`
- **Purpose:** Load KG from `kg_export.json`, build **OrgChart** (DomainBuilder clusters entities into domains with TopicSubAgents), run **QAOrchestrator** for query routing and **DomainExpertAgent** for answering from subgraphs.
- **Concepts:** Domain, TopicSubAgent, OrgChart, DomainBuilder, DomainExpertAgent, QAOrchestrator; multi-hop helpers: `find_paths`, `paths_to_text`, `neighbourhood`.

### 7.4 PDF Extraction

- **Script:** `extract_pdf.py`
- **Purpose:** Extract plain text from PDF (e.g. for `article_text.txt` or the file used in `run_pipeline_on_text.py`). Not part of the KG pipeline itself.

### 7.5 Programmatic Usage

```python
from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig, KnowledgeGraph

documents = [{"id": "doc1", "text": "...", "metadata": {}}]
orchestrator = DeliberativeOrchestrator(
    llm_config=LLMConfig(model="gemma3:27b", temperature=0.2),
    quality_threshold=0.6,
    enable_deliberation=True,
)
results = orchestrator.process_corpus(documents)
export = orchestrator.export()  # knowledge_graph, memory, stats, processing_history
# Export KG: export["knowledge_graph"] or orchestrator.knowledge_organizer.export_knowledge_graph()
```

---

## 8. Key Modules and File Map

| Path | Purpose |
|------|--------|
| `multi_agent_kg/core/deliberative_orchestrator.py` | Main pipeline runner; wires all agents and deliberation |
| `multi_agent_kg/core/memory.py` | SharedMemory, MemoryType, BlackboardEntry |
| `multi_agent_kg/core/communication.py` | MessageBus, CollaborationProtocol, AgentMessage |
| `multi_agent_kg/core/deliberation.py` | DeliberationCoordinator, Hypothesis, Vote, VoteType, DeliberationStatus |
| `multi_agent_kg/core/knowledge_graph.py` | KnowledgeGraph, Entity, Triple, Conflict |
| `multi_agent_kg/core/config.py` | LLMConfig, RelationType, RelationSchema |
| `multi_agent_kg/core/adaptive_config.py` | ModelSpec, DomainSchema, DomainTaxonomy, batching/tuning |
| `multi_agent_kg/core/kg_operations.py` | KGDiff, compute_diff, merge_kg, load_kg, save_kg, find_entity_matches |
| `multi_agent_kg/core/incremental_enrichment.py` | IncrementalEnricher, ConflictResolver |
| `multi_agent_kg/core/domain_experts.py` | Domain, TopicSubAgent, OrgChart, DomainBuilder, DomainExpertAgent, QAOrchestrator |
| `multi_agent_kg/core/messages.py` | Message, MessageType (legacy/alternate message types) |
| `multi_agent_kg/agents/base.py` | BaseAgent, AgentContext, ExtractionResult, ModelTier |
| `multi_agent_kg/agents/document_processor.py` | DocumentProcessor |
| `multi_agent_kg/agents/domain_classifier.py` | DomainClassifier |
| `multi_agent_kg/agents/entity_extractor.py` | EntityExtractor |
| `multi_agent_kg/agents/relation_extractor.py` | RelationExtractor |
| `multi_agent_kg/agents/evidence_linker.py` | EvidenceLinker |
| `multi_agent_kg/agents/extraction_validator.py` | ExtractionValidator |
| `multi_agent_kg/agents/extraction_verification_agent.py` | ExtractionVerificationAgent |
| `multi_agent_kg/agents/knowledge_organizer.py` | KnowledgeOrganizer |
| `multi_agent_kg/llm/openai_client.py` | chat_completion, chat_completion_json, get_embedding, backend selection |
| `multi_agent_kg/utils/debug_logger.py` | DebugLogger (optional pipeline_debug.log) |
| `multi_agent_kg/utils/kg_visualizer.py` | KGVisualizer (interactive HTML, static PNG; requires pyvis, networkx, matplotlib) |

---

## 9. Conventions and Development Notes

- **Python:** 3.11+. Dependencies: `openai`, `python-dotenv`, `pydantic` (see `pyproject.toml`). Install: `pip install -e .`
- **Env:** `.env` for `LLM_BACKEND`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`. `.env` is gitignored.
- **Logging:** Optional `DebugLogger` passed into orchestrator; logs to `pipeline_debug.log` (e.g. stage headers, agent messages). Verbose vote logging only if `debug_logger.verbose_votes` is True.
- **Legacy:** `legacy_agents/` contains older orchestrator and agent variants; the canonical pipeline is in `core/deliberative_orchestrator` and `agents/`.
- **Exports:** Public API is re-exported from `multi_agent_kg.core` (see `core/__init__.py`). Package root `multi_agent_kg/__init__.py` exposes KnowledgeGraph, RelationType, RelationSchema, version.

---

## 10. Summary for Agents

When editing or extending this codebase:

1. **Pipeline order** is fixed: DocumentProcessor → … → KnowledgeOrganizer; deliberation runs after EvidenceLinker and before ExtractionValidator.
2. **Context** carries document_id, text, segments, domain, entities, relations; each agent reads and optionally updates context.
3. **Shared state** lives in SharedMemory (and blackboard) and MessageBus; agents receive these in constructor.
4. **Deliberation** is orchestrated by DeliberationCoordinator; the orchestrator submits only uncertain items (0.35–0.65 confidence) and simulates votes from DomainClassifier, EvidenceLinker, RelationExtractor, ExtractionValidator as appropriate.
5. **KG updates** happen only in KnowledgeOrganizer; other agents produce entities/triples in dict form. Orphan triples (subject/object not in KG) can be audited via `get_orphan_triples()`.
6. **LLM** is always via `BaseAgent.call_llm` or `call_llm_with_self_consistency` (and the shared `chat_completion_json` in `llm/openai_client.py`); respect model tiers and backend env vars.
7. **Run scripts** assume specific input files (e.g. `gfy083_full_plaintext.txt`, `kg_export.json`); change paths in script or pass different docs when calling the orchestrator programmatically.

This file should be updated whenever the pipeline, agents, or core APIs change so that agents and developers stay aligned with the current design.

---

## 11. Testing with Ollama (server / SSH tunnel)

The pipeline is configured to run on **Ollama** by default. All model tiers currently use **gemma3:27b** (see `multi_agent_kg/llm/openai_client.py`, `multi_agent_kg/core/deliberative_orchestrator.py`). No code changes are required to use a remote Ollama server; only environment variables.

### 11.1 Ollama model in use

- **Default model:** `gemma3:27b` (all tiers: SMALL, MEDIUM, LARGE).
- **Config:** `LLMConfig(model="gemma3:27b")` in run scripts; orchestrator `model_tiers` default to gemma3:27b.
- **Requirement:** The Ollama server must have `gemma3:27b` (or the model name you pass) available. List installed models with:  
  `curl <OLLAMA_HOST>:11434/api/tags`

### 11.2 Where the pipeline runs

| Scenario | OLLAMA_BASE_URL | Notes |
|---------|------------------|--------|
| **On server / Strands (same network as Ollama)** | `http://gpu01.mind.cs.umd.edu:11434/v1` | Set in env or `.env`. The client uses OpenAI-compatible API; base URL must include `/v1`. |
| **Local dev with SSH tunnel** | default `http://localhost:11434/v1` | After tunnel is up, no change needed. |
| **Local dev, no tunnel** | N/A | Ollama is not exposed; use tunnel or run jobs on server. |

- **Backend:** `LLM_BACKEND` defaults to `ollama`. Only set `LLM_BACKEND=openai` if using OpenAI (then `OPENAI_API_KEY` is required).
- **.env example (on server / Strands):**  
  `OLLAMA_BASE_URL=http://gpu01.mind.cs.umd.edu:11434/v1`  
  (Optionally `LLM_BACKEND=ollama`; it is the default.)

### 11.3 Verifying connection

- From a machine that can reach the Ollama host (e.g. server or your machine after tunnel):
  - **List models:**  
    `curl gpu01.mind.cs.umd.edu:11434/api/tags`  
    or, via tunnel:  
    `curl localhost:11434/api/tags`
  - Confirm `gemma3:27b` (or your chosen model) appears in the JSON.
- If the pipeline fails on first LLM call, check: (1) `OLLAMA_BASE_URL` includes `:11434/v1`, (2) network/tunnel is up, (3) model name matches a tag on the server.

### 11.4 SSH tunnel (local dev)

If you run the pipeline **locally** and Ollama is only on `gpu01.mind.cs.umd.edu` (not exposed):

1. Ensure port 11434 is free locally (stop local Ollama or use another port).
2. In a terminal:  
   `ssh -L 11434:gpu01.mind.cs.umd.edu:11434 <your_id>@mind-access00.cs.umd.edu`
3. Keep the session open. In another terminal:  
   `curl localhost:11434/api/tags`  
   to confirm you see the same models as on the server.
4. Run the pipeline as usual; it uses `OLLAMA_BASE_URL=http://localhost:11434/v1` by default.

### 11.5 Running a single test

- Use `run_pipeline_on_text.py` with a small text file (or the existing `gfy083_full_plaintext.txt`). Ensure the input file path exists; the script expects `gfy083_full_plaintext.txt` by default.
- On server/Strands: set `OLLAMA_BASE_URL=http://gpu01.mind.cs.umd.edu:11434/v1` in the environment (or in `.env` in the project root) before running.
- Outputs: console logs, `kg_export.json`, and optional visualizations (`kg_interactive.html`, `kg_static.png`).
