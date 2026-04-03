# Multi-Agent KG Pipeline — Implementation Report

## Overview

This report documents all changes made across the two Claude Code sessions that completed the 10-phase multi-agent knowledge graph extraction pipeline and improved the KG Explorer UI.

---

## Phase Implementations

The pipeline was designed as a 10-stage architecture. Phases 1–5 were already complete; Phases 6–10 were implemented in these sessions.

### Phase 1 — Adaptive Planning *(new)*
**File:** `multi_agent_kg/core/adaptive_planner.py`

Pure-Python (zero external dependencies) document analyser that selects an extraction strategy before any LLM calls are made.

- **`DocumentProfile`** dataclass — captures `word_count`, `char_count`, `sentence_count`, `avg_sentence_length`, `vocabulary_diversity`, `estimated_entities` using only `re` and `string` from the standard library.
- **`PipelineStrategy`** dataclass — holds 10 fields that control every downstream pipeline parameter: `batch_size`, `max_gleanings`, `enable_triplex`, `enable_cross_document`, `enable_critic_corrector`, `max_critic_iterations`, `segment_overlap_chars`, `quality_threshold`, `reasoning`.
- **Strategy templates** — four predefined profiles:

| Strategy | Words | Batch | Gleanings | Triplex | Critic |
|----------|-------|-------|-----------|---------|--------|
| QUICK | < 2 000 | 8 | 0 | ✗ | ✗ |
| STANDARD | 2 000–8 000 | 5 | 1 | ✗ | ✓ |
| THOROUGH | > 8 000 | 3 | 2 | ✓ | ✓ |
| MULTI_DOC | corpus | 5 | 1 | ✗ | ✓ |

- **Adjustments** — strategy parameters are dynamically modified based on: vocabulary diversity, average sentence length, entity density, and domain signals (biomedical, legal, financial, technical keywords).
- **Why it improves performance** — avoids over-processing short docs (QUICK = ~6× fewer LLM calls) and enables full multi-pass extraction only when the document warrants it.

---

### Phase 6 — Triplex Parallel Extraction *(wired)*
**File:** `multi_agent_kg/agents/triplex_extractor.py` *(pre-existing, now integrated)*

The SciPhi Triplex model (3.8B params, schema-guided extraction) was previously implemented but not called from the orchestrator.

- Lazy model-availability check via `GET /api/tags` on Ollama — returns empty results gracefully if model is absent.
- Runs **in parallel** (same thread, sequential for simplicity) to the main LLM pipeline to produce an independent set of triples.
- Results tagged `"source": "triplex"` for downstream merging.

---

### Phase 6 — Schema Alignment *(new)*
**File:** `multi_agent_kg/agents/schema_aligner.py`

Merges and normalises extraction results from the main LLM pipeline and Triplex.

- **Entity deduplication** — fuzzy text matching (rapidfuzz `token_sort_ratio` with Jaccard fallback, threshold 0.85). Higher-confidence entry wins; aliases merged; `sources` list tracks provenance.
- **Triple deduplication** — case-insensitive `(subject, relation, object)` key indexing. Matching triples from both sources get a +0.10 confidence boost; evidence lists are merged.
- **Type normalisation** — `_to_upper_snake_case()` converts all entity types and relation names (camelCase, PascalCase, hyphenated) to `UPPER_SNAKE_CASE`.
- **Ambiguous type resolution** — when two sources assign different types to the same entity, an optional LLM call picks the best type with a one-sentence reason; fallback is the first type.
- Participates in the deliberation framework via `evaluate_hypothesis_for_vote()`.

---

### Phase 7 — Deep Agents Integration *(new)*
**Files:** `multi_agent_kg/deep_agent/__init__.py`, `multi_agent_kg/deep_agent/agent.py`

Optional LangChain Deep Agents wrapper. Falls back transparently to the standard pipeline if `deepagents` is not installed.

- **`DeepAgentOrchestrator`** — wraps `DeliberativeOrchestrator` with a Deep Agents graph:
  - SubAgent pattern: `extraction_worker` + `validation_worker` for context-isolated stage execution.
  - Middleware stack: model retry (3×, 1.5× backoff), tool retry (2×), conversation summarisation.
  - Filesystem-based checkpointing: each pipeline stage writes intermediate JSON to a configurable workspace directory (`/tmp/kg_workspace` by default).
  - LangGraph checkpoint directory for crash recovery and stage resumption.
- **6 LangChain tools** wrapping pipeline stages: `segment_document`, `classify_domain`, `extract_entities`, `extract_relations`, `validate_extractions`, `build_knowledge_graph`.
- **Graceful degradation** — any import failure, build failure, or invocation error falls back to the standard `DeliberativeOrchestrator` path.

---

### Phase 8 — (renumbered) Entity Resolution *(pre-existing)*
KGGen-style fuzzy matching + LLM clustering with union-find for global entity deduplication. Was already implemented; renumbered to Stage 6 in the 10-stage pipeline.

---

### Phase 9 — Critic-Corrector Verification *(pre-existing, wired as Stage 9)*
FinReflectKG-style iterative verification loop. Was already implemented; integrated as Stage 9 of the 10-stage pipeline.

- Runs up to `max_critic_iterations` rounds (set by `PipelineStrategy`).
- Critic agent identifies issues → Corrector agent applies targeted fixes → repeat.
- Early exit when critic finds no issues.

---

## Orchestrator Changes

**File:** `multi_agent_kg/core/deliberative_orchestrator.py`

The orchestrator was expanded from an 8-stage to a 10-stage pipeline.

### New imports
```python
from multi_agent_kg.core.adaptive_planner import AdaptivePlanner

# Optional (graceful degradation if not available)
try: from multi_agent_kg.agents.triplex_extractor import TriplexExtractor
try: from multi_agent_kg.agents.schema_aligner import SchemaAligner
```

### Updated `__init__`
- `self.adaptive_planner = AdaptivePlanner()` — instantiated at startup.
- `self.enable_triplex = TRIPLEX_AVAILABLE` — auto-detected.
- `self.triplex_extractor` and `self.schema_aligner` — initialised in `_init_agents`.
- `PipelineProgress(total_stages=10)` — progress bar updated from 8 to 10 stages.

### Updated `process_document` — 10-stage pipeline

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1 | AdaptivePlanner | Profiles document → selects QUICK/STANDARD/THOROUGH/MULTI_DOC strategy |
| 2 | DocumentProcessor | Segments text with configurable overlap |
| 3 | DomainClassifier | Discovers domain and schema (≤7 entity types, ≤15 relations) |
| 4 | FastExtractor (optional) | GLiNER/GLiREL zero-cost baseline extraction |
| 5 | EntityExtractor | Consolidated + gleaning passes |
| 6 | EntityResolver | KGGen clustering + canonicalisation |
| 7 | RelationExtractor | Consolidated + gleaning; populates RelationLibrary |
| 8 | Triplex + SchemaAligner (optional) | Parallel extraction merge |
| 9 | CriticCorrectorLoop | Iterative verification (strategy-controlled iterations) |
| 10 | KnowledgeOrganizer | Final KG integration |

### Strategy-driven execution
The `PipelineStrategy` from Stage 1 controls:
- `enable_triplex` → whether Stage 8 runs
- `max_critic_iterations` → how many critic-corrector loops in Stage 9
- `max_gleanings` → gleaning passes in Stages 5 and 7

---

## Package Export Updates

**`multi_agent_kg/agents/__init__.py`** — added exports:
- `EntityResolver`, `CriticAgent`, `CorrectorAgent`, `TriplexExtractor`, `SchemaAligner`

**`multi_agent_kg/core/__init__.py`** — added exports:
- `AdaptivePlanner`, `PipelineStrategy`, `DocumentProfile`

---

## Progress Display

**File:** `multi_agent_kg/utils/progress.py`

`PIPELINE_STAGES` list updated from 8 to 10 entries:
```
1. Adaptive Planning
2. Document Processing
3. Domain Classification
4. Fast First Pass (GLiNER/GLiREL)
5. Entity Extraction
6. Entity Resolution
7. Relation Extraction
8. Triplex + Schema Alignment
9. Critic-Corrector Verification
10. Knowledge Graph Integration
```

---

## KG Explorer UI Improvements

**File:** `kg_explorer.html`

### Problem 1: Cramped nodes
**Root cause:** `options` had no physics engine — all node positions depended on pre-computed x/y coordinates embedded in the data.

**Fix:** Added `barnesHut` physics with repulsion and spring forces:
```javascript
physics: {
  barnesHut: {
    gravitationalConstant: -18000,
    centralGravity: 0.2,
    springLength: 180,
    avoidOverlap: 0.4
  },
  stabilization: { iterations: 350, fit: true }
}
```
Also added `interaction.hideEdgesOnDrag` for performance on large graphs.

### Problem 2: QA result nodes/edges not highlighted
**Root cause 1:** All four pre-loaded `qaData` cards had `evidence_triples: []` — the highlighting code was correct but had no data to highlight.

**Root cause 2:** `askQuestion()` (live QA server) created a new card but never called `highlightEvidence()` — it just inserted raw HTML text without registering the result in `qaData`.

**Fix:** `askQuestion()` now:
1. Pushes the server result (including `evidence_triples`) into `qaData` as a new entry.
2. Auto-activates the new card and immediately calls `highlightEvidence(newQi)`.
3. Builds evidence list HTML with matched/unmatched indicators.

### Problem 3: Highlight didn't focus on the path
**Fix:** `highlightEvidence()` now calls `network.fit({ nodes: highlightedNodeIds, animation: ... })` to zoom into the highlighted subgraph after applying colours.

### New UI features

| Feature | Description |
|---------|-------------|
| **Graph toolbar** | Fit / Freeze / Clear buttons in top-right of graph panel |
| **Physics toggle** | "Freeze" button pauses physics; "Unfreeze" resumes — useful once layout stabilises |
| **Stabilisation indicator** | Animated "Spreading nodes… N%" banner while physics runs, disappears when done |
| **Node info panel** | Click any node to see its name, type, and connection count in the QA panel |
| **Node search** | Search box filters the graph by node label — matching nodes highlighted green, others dimmed |
| **Path summary badge** | Shows "✓ 3/4 triples highlighted" or a warning when evidence isn't found |
| **Score bar** | Thin progress bar under each QA card showing confidence visually |
| **Live query cards** | Dynamic QA results are now full interactive cards with evidence list and graph linking |
| **Custom scrollbar** | Thin custom scrollbar on QA panel |
| **Edge arrows** | Small directional arrows on edges show relation direction |

---

## Summary of Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `multi_agent_kg/core/adaptive_planner.py` | **Created** | Phase 8: Document profiling + strategy selection |
| `multi_agent_kg/agents/schema_aligner.py` | **Created** | Phase 6: Merge + normalise multi-source extractions |
| `multi_agent_kg/deep_agent/__init__.py` | **Created** | Phase 7: Package init |
| `multi_agent_kg/deep_agent/agent.py` | **Created** | Phase 7: LangChain Deep Agents wrapper |
| `multi_agent_kg/core/deliberative_orchestrator.py` | **Modified** | Wired all 10 stages; added adaptive planning + triplex |
| `multi_agent_kg/utils/progress.py` | **Modified** | Updated stage list from 8 to 10 |
| `multi_agent_kg/agents/__init__.py` | **Modified** | Exported new agents |
| `multi_agent_kg/core/__init__.py` | **Modified** | Exported AdaptivePlanner, PipelineStrategy, DocumentProfile |
| `kg_explorer.html` | **Modified** | Fixed physics, QA highlighting, added toolbar + search |
