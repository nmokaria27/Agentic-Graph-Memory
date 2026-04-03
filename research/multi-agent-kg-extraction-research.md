# Deep Research: Multi-Agent Knowledge Graph Extraction from Text

**Date**: 2026-03-28
**Purpose**: Survey of existing projects, frameworks, and research that improve multi-agent KG extraction from text.

---

## Table of Contents

1. [GraphRAG (Microsoft)](#1-graphrag-microsoft)
2. [LightRAG (HKUDS)](#2-lightrag-hkuds)
3. [REBEL (Babelscape)](#3-rebel-babelscape)
4. [LangChain LLMGraphTransformer](#4-langchain-llmgraphtransformer)
5. [Multi-Agent KG Systems (KARMA, Agentic-KGR)](#5-multi-agent-kg-systems)
6. [Triplex (SciPhi)](#6-triplex-sciphi)
7. [GLiNER / GLiREL](#7-gliner--glirel)
8. [NuExtract (NuMind)](#8-nuextract-numind)
9. [KGQA Systems](#9-kgqa-systems)
10. [Structured Extraction with Constrained Decoding](#10-structured-extraction-with-constrained-decoding)
11. [Supplementary: iText2KG and Production Best Practices](#11-supplementary-itext2kg-and-production-best-practices)
12. [Synthesis: Applicability to Multi-Agent KG Pipeline](#12-synthesis-applicability-to-multi-agent-kg-pipeline)

---

## 1. GraphRAG (Microsoft)

**GitHub**: https://github.com/microsoft/graphrag (~31.6k stars)
**Docs**: https://microsoft.github.io/graphrag/
**Paper**: "From Local to Global: A GraphRAG Approach to Query-Focused Summarization" (2024)

### What It Does

GraphRAG is a modular graph-based Retrieval-Augmented Generation system that transforms unstructured documents into structured knowledge graphs and uses hierarchical community detection for querying. It solves the fundamental limitation of traditional vector-based RAG: inability to answer questions requiring understanding across an entire dataset ("global queries").

### Core Approach: The 6-Phase Indexing Pipeline

**Phase 1 - Compose TextUnits**: Segment input documents into analyzable chunks (default 1200 tokens with configurable overlap). Larger chunks trade fidelity for speed.

**Phase 2 - Document Processing**: Create provenance links between original documents and derived TextUnits for source attribution and breadcrumb tracking.

**Phase 3 - Graph Extraction**:
- LLM extracts entities (title, type, description) and relationships (source, target, description) from each TextUnit
- Entities with the same title/type are merged by concatenating their descriptions into arrays
- A second LLM pass distills multiple descriptions into a single summary per entity/relationship
- Optional claim extraction identifies factual statements with evaluated status and time-bounds (disabled by default)

**Phase 4 - Graph Augmentation**: Applies the **Hierarchical Leiden Algorithm** for community detection. Recursively clusters the graph into communities until reaching size thresholds. Produces a hierarchical community structure.

**Phase 5 - Community Summarization**: LLM generates reports for each community including executive overview, key entities/relationships/claims. Then creates condensed versions for efficient reference.

**Phase 6 - Text Embedding**: Generates embeddings for TextUnits, entity/relationship descriptions, and community reports. Stored in configured vector store.

### Key Innovations

- **Hierarchical community detection**: Uses Leiden algorithm to create multi-level summaries of the knowledge graph, enabling both local (entity-focused) and global (theme-focused) queries
- **Community reports**: LLM-generated summaries at multiple abstraction levels solve the "needle in a haystack" problem for global queries
- **Gleaning**: The `entity_extract_max_gleaning` parameter controls iterative extraction loops, appending history messages to catch missed entities
- **Prompt tuning**: Dedicated prompt tuning pipeline to adapt extraction to specific domains

### Results

- 70-80% win rate on comprehensiveness vs. traditional RAG in human evaluations
- Excels at global queries ("What are the top themes in this dataset?") where traditional RAG fails
- High LLM cost due to multiple passes (extraction, summarization, community reports, embedding)

### Works with Local/Open-Source Models

Yes, through OpenAI-compatible API endpoints. Can use any LLM that supports the OpenAI chat completions format. However, entity extraction quality depends heavily on model capability -- smaller models may miss entities or produce lower-quality descriptions.

### Applicability to Multi-Agent KG Pipeline

- The **gleaning** technique (iterative extraction with history) directly addresses the "missed connections" problem
- The **community detection + hierarchical summarization** approach provides a powerful querying mechanism
- The **entity merging** strategy (concatenate descriptions, then summarize) is a practical deduplication approach
- The 6-phase pipeline architecture is a good reference for structuring a multi-agent extraction workflow
- The prompt tuning system could be adapted for domain-adaptive extraction

---

## 2. LightRAG (HKUDS)

**GitHub**: https://github.com/HKUDS/LightRAG (~23k+ stars)
**Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025, arXiv:2410.05779)
**PyPI**: `lightrag-hku`

### What It Does

LightRAG is a lightweight alternative to GraphRAG that integrates graph-based text indexing with a dual-level retrieval framework. It builds a knowledge graph from documents and uses both local (entity-level) and global (relationship-level) retrieval for answering queries.

### Core Approach

**Indexing**:
- Segments documents into chunks
- Uses LLM to extract entities and relationships, building a comprehensive knowledge graph
- Supports multiple graph backends: NetworkX, Neo4j, PostgreSQL, AGE, OpenSearch
- Supports multiple vector backends: NanoVectorDB, PGVector, Milvus, Chroma, Faiss, Qdrant, MongoDB

**Retrieval** (Dual-Level):
- **Local mode**: Entity-focused, context-dependent information retrieval
- **Global mode**: Relationship-focused, utilizing global knowledge
- **Hybrid mode**: Combines local and global
- **Mix mode** (recommended): Integrates knowledge graph traversal and vector retrieval with reranking
- **Naive mode**: Basic vector search (baseline)

**Key Configuration Parameters**:
- `entity_extract_max_gleaning`: Same iterative extraction as GraphRAG
- `chunk_token_size`: 1200 tokens default
- `top_k`: 60 for retrieval
- Supports reranker models (BAAI/bge-reranker-v2-m3 recommended)

### Key Innovations

- **Dramatically simpler than GraphRAG**: No community detection step, no hierarchical summarization
- **Incremental updates**: Supports adding documents without full reprocessing
- **Document deletion**: With automatic KG regeneration
- **Reranker integration**: Significantly boosts retrieval quality
- **Citation support**: Source attribution for answers
- **65-80% cost savings** vs. GraphRAG
- **10x token reduction** with <5% accuracy loss

### Results/Benchmarks

- Outperforms NaiveRAG, GraphRAG, HyDE, and RQ-RAG across Comprehensiveness, Diversity, Empowerment, and Overall metrics
- ~30% reduction in query latency (~80ms vs ~120ms for standard RAG)
- Published at EMNLP 2025

### Works with Local/Open-Source Models

Yes, excellent support. Explicitly supports Ollama and any OpenAI-compatible API. Recommends 32B+ parameter models with 32K+ context for extraction quality. Specifically notes improved accuracy for open-source LLMs like Qwen3-30B-A3B in recent updates.

### Applicability to Multi-Agent KG Pipeline

- The **dual-level retrieval** (local + global) is directly applicable for QA over extracted KGs
- **Incremental construction** solves the problem of processing new documents without rebuilding
- Much **simpler architecture** than GraphRAG -- good starting point
- The **reranker integration** pattern could improve answer quality in QA pipelines
- Entity/relationship extraction prompts could be studied for improving extraction quality
- Mix mode combining KG traversal and vector search is a strong querying strategy

---

## 3. REBEL (Babelscape)

**GitHub**: https://github.com/Babelscape/rebel
**HuggingFace**: https://huggingface.co/Babelscape/rebel-large
**Paper**: "REBEL: Relation Extraction By End-to-end Language generation" (EMNLP 2021 Findings)

### What It Does

REBEL reframes relation extraction as a sequence-to-sequence generation task. Instead of separate NER and RE pipelines, it generates triplets (head, relation, tail) in a single autoregressive pass using a fine-tuned BART model.

### Core Approach

**Architecture**: Fine-tuned BART (seq2seq transformer) that generates linearized triplets.

**Linearization Format**: Uses special tokens `<triplet>`, `<subj>`, `<obj>` to delimit entities and relations in the output sequence. Example output:
```
<triplet> Punta Cana <subj> Dominican Republic <obj> country <triplet> Punta Cana <subj> Higuey <obj> located in
```

**Models Available**:
- `Babelscape/rebel-large`: English, 200+ relation types from Wikidata
- `Babelscape/mrebel-large`: Multilingual (17 languages), 400 relation types
- `Babelscape/mrebel-large-32`: Multilingual, 32 relation types (filtered)

**Dataset**: CROCODILE (automatiC RelatiOn extraCtiOn Dataset wIth nLi filtEring) -- silver-standard dataset derived from Wikidata with NLI filtering.

### Key Innovations

- **End-to-end approach**: No separate NER + RE pipeline, eliminating error propagation
- **Seq2seq framing**: Uses autoregressive generation to produce triplets, sidestepping fixed relation type limitations
- **200+ relation types**: Much broader coverage than traditional RE models
- **spaCy integration**: Can be used as a spaCy pipeline component for seamless integration
- **Multilingual support** via mREBEL

### Results/Benchmarks

State-of-the-art on multiple RE benchmarks at time of publication:
- NYT (Relation Extraction)
- CONLL04 (Joint Entity and Relation Extraction)
- ADE Corpus (Relation Extraction)
- Re-TACRED (Relation Extraction)

### Works with Local/Open-Source Models

Yes -- REBEL is a standalone model (~400M parameters for the large version). Runs locally on GPU with HuggingFace Transformers. No API calls needed.

### Limitations

- Fixed to Wikidata relation types (200 for English, 400 for multilingual) -- cannot extract novel relation types without fine-tuning
- Model is from 2021, predates modern LLM-based approaches
- Performance on domain-specific text (e.g., scientific) may be limited
- License: CC BY-SA-NC 4.0 (non-commercial)

### Applicability to Multi-Agent KG Pipeline

- Could serve as a **fast first-pass extraction** agent that produces triplets cheaply before LLM-based verification
- The **spaCy integration** makes it easy to add to existing NLP pipelines
- **Zero API cost** for extraction -- runs entirely locally
- Limited to Wikidata relation types, which may not cover domain-specific needs
- Could be combined with LLM-based extraction: REBEL for high-confidence common relations, LLM for domain-specific or novel ones
- The **mREBEL multilingual** variant enables cross-language KG construction

---

## 4. LangChain LLMGraphTransformer

**Docs**: https://python.langchain.com/api_reference/experimental/graph_transformers/
**Source**: https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py
**Blog**: https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/

### What It Does

LLMGraphTransformer is a LangChain module that uses any LLM to extract entities and relationships from text, producing graph documents suitable for storage in Neo4j or other graph databases.

### Core Approach

**Two Operational Modes**:

1. **Tool-Based Mode** (default): Uses `llm.with_structured_output(schema)` for native function calling. The LLM returns validated Pydantic objects directly. Supports node/relationship property extraction. This is the preferred mode when the LLM supports structured output.

2. **Prompt-Based Mode** (fallback): Activates when the LLM lacks structured output support. Returns unstructured text requiring JSON parsing (uses `json-repair` package for malformed JSON). Does not support property extraction.

**Schema Definition**:
- Uses dynamically generated Pydantic models (`SimpleNode`, `SimpleRelationship`)
- Core container model `_Graph` holds optional lists of nodes and relationships
- `UnstructuredRelation` model defines: head, head_type, relation, tail, tail_type

**Configurable Parameters**:
- `allowed_nodes`: List of entity types the LLM should extract (e.g., ["Person", "Organization"])
- `allowed_relationships`: Either string list (e.g., ["WORKS_FOR"]) or tuple list with schema constraints (e.g., [("Person", "WORKS_FOR", "Company")])
- `node_properties`: Properties to extract per node
- `relationship_properties`: Properties to extract per relationship

**Entity Normalization**: Automatically titles node IDs, capitalizes types, converts relationship types to UPPERCASE_WITH_UNDERSCORES.

**System Prompt**: Instructs the LLM to maintain entity consistency, avoid specific/momentary relationship types, use elementary node labels, and perform coreference resolution (ensuring "John Doe", "he", "the physicist" resolve to the same entity).

### Also Available: GlinerGraphTransformer

LangChain also provides a `GlinerGraphTransformer` that uses GLiNER + GLiREL models instead of an LLM for extraction, integrated into the same graph transformer interface.

### Key Innovations

- **Schema-constrained extraction**: Tuple-based relationship specifications enforce which entity types can connect
- **Dual-mode flexibility**: Works with both structured-output-capable and basic LLMs
- **Coreference resolution** built into the system prompt
- **Property extraction**: Can extract attributes of entities and relationships, not just types

### Works with Local/Open-Source Models

Yes. Tool-based mode requires models with function calling / structured output (e.g., Llama 3.1+ with tool use). Prompt-based mode works with any LLM but is less reliable.

### Applicability to Multi-Agent KG Pipeline

- **Schema-constrained extraction** is directly valuable -- defining allowed entity types and relationship constraints reduces hallucination
- The **tuple-based relationship specification** (e.g., only allow "Person" -> "WORKS_FOR" -> "Company") could be adapted for domain-specific schemas
- The **coreference resolution prompt** is a practical technique to reduce entity fragmentation
- The **GlinerGraphTransformer** variant provides a zero-LLM-cost extraction option
- The dual-mode approach shows how to handle models with and without structured output capabilities
- Could serve as the extraction backbone, with additional agents for verification and deduplication

---

## 5. Multi-Agent KG Systems

### 5a. KARMA (NeurIPS 2025 Spotlight)

**Paper**: "KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment" (arXiv:2502.06472)
**OpenReview**: https://openreview.net/forum?id=k0wyi4cOGy
**Blog implementation**: https://atalupadhyay.wordpress.com/2025/03/12/building-a-multi-agent-system-for-knowledge-graph-enrichment-implementing-karma/

#### What It Does

KARMA is a multi-agent LLM framework that automatically updates and expands knowledge graphs from scientific papers. It is the most directly relevant existing work to a multi-agent KG extraction pipeline.

#### Architecture: 9 Collaborative Agents

All orchestrated by a **Central Controller Agent (CCA)**:

1. **Ingestion Agents (IA)**: Document retrieval and normalization
2. **Reader Agents (RA)**: Filter and parse documents into segments
3. **Summarizer Agents (SA)**: Create concise summaries from segments
4. **Entity Extraction Agents (EEA)**: Extract entities from text
5. **Relationship Extraction Agents (REA)**: Extract relations between entities
6. **Schema Alignment Agents (SAA)**: Align extracted knowledge to domain-specific schema
7. **Conflict Resolution Agents (CRA)**: Resolve contradictions and merge duplicates
8. **Evaluator Agents (EA)**: Verify extracted knowledge quality

Agents operate in parallel where possible, with the CCA managing dependencies and data flow.

#### Key Results

Tested on **1,200 PubMed articles** across three biomedical domains:
- Identified up to **38,230 new entities**
- Achieved **83.1% LLM-verified correctness**
- Reduced **conflict edges by 18.6%** through multi-layer assessments

#### Key Innovations

- **Schema alignment agents** ensure extracted knowledge conforms to domain ontology
- **Conflict resolution agents** detect and resolve contradictions between different extractions
- **Multi-layer assessment** (extraction -> alignment -> conflict resolution -> evaluation) catches errors at multiple stages
- **Parallel agent execution** enables redundancy and robustness

#### Applicability to Multi-Agent KG Pipeline

This is the most directly comparable system. Key architectural takeaways:
- **Separate agents for schema alignment and conflict resolution** -- these are distinct from extraction
- **The evaluator agent** acts as a final quality gate
- **Multi-layer assessment** significantly reduces errors
- The pipeline structure (ingest -> parse -> summarize -> extract -> align -> resolve -> evaluate) closely mirrors a well-designed multi-agent KG pipeline

---

### 5b. Agentic-KGR (arXiv 2025)

**Paper**: "Agentic-KGR: Co-evolutionary Knowledge Graph Construction through Multi-Agent Reinforcement Learning" (arXiv:2510.09156)

#### What It Does

Agentic-KGR enables co-evolution between LLMs and knowledge graphs through multi-round reinforcement learning. The LLM and KG improve each other iteratively.

#### Architecture

Models the interaction as a POMDP (Partially Observable Markov Decision Process). Agents interact with a comprehensive tool pool for KG operations and receive dual rewards.

**Three Key Innovations**:

1. **Dynamic Schema Expansion**: Systematically extends graph ontologies beyond predefined boundaries during training. Agents monitor and expand schemas as new entity/relation types emerge.

2. **Retrieval-Augmented Memory System**: Bidirectional feedback loop -- improved extraction produces richer graphs, which enable better retrieval for future decisions.

3. **Multi-Scale Prompt Compression**: Learnable compression preserving critical information while reducing computational complexity.

#### Reinforcement Learning Design

**Dual Reward Structure**:
- **Environmental Reward**: Coverage gain (submodular function measuring neighborhood expansion) + von Neumann entropy gain (structural diversity)
- **Task Reward**: Extraction accuracy + tool usage efficiency
- **Adaptive mixing**: Mirror descent on Pareto frontier

**Specific rewards**: +0.05 for successful tool calls, -0.1 for failures, +1.5 for full match accuracy, +-1.0 for format compliance.

#### Results

- **+33.3 points improvement** over existing RL methods in graph extraction
- **+12.8 points** in downstream QA tasks
- ConfigKG RE: 72.50 (Agentic) vs 60.28 (baseline) with Qwen2.5 32B
- WirelessKG RE: 53.92 vs 45.18 (baseline)
- Downstream QA with GraphRAG: up to 98.46 accuracy (MA5600T domain)

#### Applicability to Multi-Agent KG Pipeline

- The **co-evolution** concept is powerful: the KG and extraction quality improve together over time
- **Dynamic schema expansion** addresses the problem of unknown entity/relation types in new domains
- The **RL reward design** provides a principled way to balance graph quality metrics
- However, requires RL training infrastructure which adds significant complexity
- Best suited for ongoing, evolving KG construction rather than one-time extraction

---

### 5c. Collaborative Multi-Agent LLM Approach

**Source**: Preprint (2025)

Uses 5 specialized agents: query generator, domain model generator, domain model populator, knowledge graph agent, plus a coordinator. Key innovation is intermediate layers -- question generation and domain model creation steps enable more meaningful relationships, producing hierarchically connected graphs rather than isolated nodes.

---

## 6. Triplex (SciPhi)

**HuggingFace**: https://huggingface.co/SciPhi/Triplex
**Ollama**: `sciphi/triplex`
**Blog**: https://www.sciphi.ai/blog/triplex
**GGUF**: https://huggingface.co/QuantFactory/Triplex-GGUF

### What It Does

Triplex is a fine-tuned Phi3-3.8B model specifically optimized for knowledge graph triplet extraction. It takes text, entity types, and predicates as input and produces structured triplets.

### Core Approach

**Architecture**: Based on Phi3-3.8B (4B parameters, BF16 tensor type)

**Training Methodology**:
- Further trained an SFT model with a preference-based dataset
- Dataset created from **majority voting** and **topological sorting**
- Trained using **DPO (Direct Preference Optimization)** and **KTO (Kahneman-Tversky Optimization)**

**Input Format**:
```
Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text...
Entity Types: {entity_types}
Predicates: {predicates}
Text: {text}
```

Users specify entity types and predicates (allowed relation types), making extraction schema-guided.

### Key Innovations

- **Schema-guided extraction**: Users define entity types and predicates, constraining output
- **98% cost reduction** vs. GPT-4 for KG construction
- **Outperforms GPT-4** at KG extraction tasks at 1/60th the cost
- **Runs locally**: Available on Ollama and as GGUF quantization
- **Small model**: Only 3.8B parameters, feasible for local deployment

### License

CC-BY-NC-SA-4.0 (non-commercial), but waived for organizations under $5M USD revenue.

### Applicability to Multi-Agent KG Pipeline

- **Ideal for local deployment** as a dedicated extraction agent -- small enough to run alongside other models
- **Schema-guided extraction** (entity types + predicates) aligns perfectly with domain-adaptive extraction
- Could serve as a **fast, cheap extraction agent** complementing larger LLM verification agents
- Available on **Ollama** -- direct integration with existing Ollama-based infrastructure
- The DPO/KTO training methodology could inform fine-tuning custom extraction models
- **Limitation**: Non-commercial license may restrict some use cases

---

## 7. GLiNER / GLiREL

### GLiNER

**GitHub**: https://github.com/urchade/GLiNER (~3,000+ stars)
**Paper**: NAACL 2024
**Models**: https://huggingface.co/urchade/gliner_medium-v2.1

#### What It Does

GLiNER is a zero-shot Named Entity Recognition framework using bidirectional transformer encoders. It extracts arbitrary entity types from text based on natural language labels, without requiring task-specific training.

#### Architecture

Uses a **bidirectional transformer encoder** (DeBERTa-based) that enables **parallel entity extraction** -- much faster than sequential LLM generation. The model jointly processes entity type labels and input text, computing similarity between token representations and label representations.

#### Key Features

- **Zero-shot**: Provide entity type labels in natural language, extract without fine-tuning
- **CPU-friendly**: Runs on consumer hardware, no GPU required
- **Fine-tunable**: Can adapt to specific domains with small datasets
- **Competitive with ChatGPT** for NER tasks at a fraction of the cost
- **205M parameters** (GLiNER2 multi-task version)

---

### GLiREL

**GitHub**: https://github.com/jackboyla/GLiREL (262 stars)
**Paper**: "GLiREL -- Generalist Model for Zero-Shot Relation Extraction" (NAACL 2025, arXiv:2501.03172)
**PyPI**: `glirel`

#### What It Does

GLiREL extends the GLiNER approach to zero-shot relation extraction. It classifies relationships between entity pairs using natural language relation labels, without having seen those relation types during training.

#### Architecture (Detailed)

Three core components:
1. **Pre-trained bidirectional encoder** (DeBERTa V3-large, 467M parameters): Jointly processes relation labels and input text
2. **Entity pair representation module**: `e_ij = FFN(h_i || h_j)`, then pair representation `kappa_uv = FFN(e_u || e_v)`
3. **Scoring module**: `phi(u,v,t) = sigma(kappa_uv^T * q_t)` -- sigmoid-based similarity scoring

**Key design**: Processes multiple entity pairs and relation types in a **single forward pass**, unlike approaches requiring separate inputs per pair-label combination.

**Optional refinement layers**: Cross-attention and self-attention mechanisms (max 2 layers) allow entity pairs to attend to relation types.

#### Training

- Uses **synthetic pretraining** with dataset generated by Mistral 7B-Instruct-v0.3 on Fineweb data
- Resulting dataset: **63,493 texts with 25,619,624 annotated relations**
- Training: AdamW, batch size 8, 20,000 steps on single Tesla T4 GPU
- Regularization: Random label dropping, label shuffling

#### Benchmark Results

| Benchmark | GLiREL F1 | Best Baseline F1 | GPT-4o F1 |
|-----------|-----------|-------------------|-----------|
| FewRel (m=5) | **94.20** | 93.62 (TMC-BERT) | 89.20 |
| WikiZSL (m=5) | **83.28** | 78.97 (TMC-BERT) | 72.41 |

**Inference speed** (m=10, batch 32):
- GLiREL: 47.60 sent/s (WikiZSL GPU)
- TMC-BERT: 1.41 sent/s
- RelationPrompt: 2.06 sent/s

**20x faster** than competitors on WikiZSL, while achieving higher accuracy.

#### Entity Type Constraints

Supports defining which entity types are allowed for relation heads and tails:
```python
labels = {
    "glirel_labels": {
        "founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "headquartered_in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE"]}
    }
}
```

---

### GLiNER2 (Multi-Task)

**GitHub**: https://github.com/fastino-ai/GLiNER2
**Paper**: EMNLP 2025 System Demo

Unifies NER, Text Classification, Structured Data Extraction, and Relation Extraction into a **single 205M parameter model** with a schema-driven interface. Executes multiple extraction tasks in a single inference call.

---

### Combined GLiNER + GLiREL Pipeline

LangChain provides a `GlinerGraphTransformer` that combines both models:
1. GLiNER extracts entities
2. GLiREL classifies relations between extracted entities
3. Output is formatted as graph documents

Also integrates with spaCy pipelines as custom components.

### Applicability to Multi-Agent KG Pipeline

- **Zero LLM cost** for entity and relation extraction -- runs on CPU
- **20x faster** than alternatives, enabling real-time extraction
- **Zero-shot** means no training needed for new domains -- just provide labels
- **Schema-constrained** with entity type constraints for relations
- **Ideal as a fast first-pass agent**: GLiNER+GLiREL extract candidates quickly, then LLM agents verify/enrich
- **Complementary to LLM extraction**: Catches structured patterns LLMs might miss, while LLMs handle nuance
- **Limitation**: 512-token sequence length ceiling (DeBERTa limit)
- **Limitation**: Performance depends on label quality -- labels must be descriptive
- The **synthetic dataset generation** protocol could be used to create domain-specific training data

---

## 8. NuExtract (NuMind)

**HuggingFace**: https://huggingface.co/numind/NuExtract-2.0-8B
**Blog**: https://numind.ai/blog/outclassing-frontier-llms----nuextract-2-0-takes-the-lead-in-information-extraction
**Platform**: https://nuextract.ai

### What It Does

NuExtract is a family of models specifically trained for structured information extraction. Given text and a JSON template describing what to extract, it produces structured JSON output. NuExtract 2.0 supports multimodal inputs (vision) and abstraction capabilities.

### Core Approach

**Architecture**: Based on Qwen vision-language models:
- **NuExtract-2.0-2B**: 2.2B params, Qwen 2.0 VL base, MIT license
- **NuExtract-2.0-4B**: 3.75B params, Qwen 2.5 VL base, research license
- **NuExtract-2.0-8B**: 8.3B params, Qwen 2.5 VL base, MIT license
- **NuExtract 2.0 PRO**: Larger proprietary model via API

**Input**: Text + JSON template describing desired fields
**Output**: Filled JSON matching the template

**Template Types**:
- `"verbatim-string"`: Pure extractive (text must appear verbatim in source)
- `"choice"`: Classification into predefined categories
- Standard types for abstraction/reformatting

### Key Innovations

- **Purely extractive mode**: All output text is present verbatim in the source (reduces hallucination)
- **Template-driven**: Define schema as JSON, model fills it in
- **In-context learning**: Adding 3 examples improves PRO by +6 F-Score points
- **Multimodal**: Processes PDFs, scanned documents, receipts as images
- **Higher precision than recall**: Prioritizes accuracy over completeness

### Results/Benchmarks

| Model | F-Score |
|-------|---------|
| NuExtract 2.0 PRO | **~82** |
| GPT-4.1 | ~73 (-9 pts) |
| Claude 4 Opus | ~77 (-5 pts) |
| o3 (reasoning) | ~79 (-3 pts) |
| Gemini 2.5 PRO | ~80 (-2 pts) |
| NuExtract-2.0-8B (open) | ~73 |

### Works with Local/Open-Source Models

Yes. The 2B, 4B, and 8B models are available on HuggingFace with GGUF quantizations for local deployment. 32K token context window.

### Limitations

- 32K context (~60 text pages)
- Struggles with extremely long extraction lists
- Occasional "laziness" on complex templates with missing data

### Applicability to Multi-Agent KG Pipeline

- **Template-driven extraction** is a powerful paradigm: define what you want, model fills it in
- The **"verbatim-string" extraction** mode eliminates hallucination for entity names
- Could serve as a **structured extraction agent** that extracts entities/relations into a predefined schema
- The **in-context learning** feature (3 examples for +6 F-Score) is practical for domain adaptation
- **GGUF quantizations** enable local deployment alongside other models
- The template approach could be combined with dynamic schema generation from domain analysis
- Higher precision (fewer false positives) is preferable for KG construction where wrong connections are worse than missed ones

---

## 9. KGQA Systems

### Recent Approaches (2025)

#### DRKG: LLM-Guided Reasoning Plans

**Paper**: "DRKG: Faithful and Interpretable Multi-Hop KGQA via LLM-Guided Reasoning Plans" (2025)

**Approach**: Plan-Retrieve-Reason paradigm:
1. LLM generates relational paths as reasoning plans
2. Relevant KG subgraphs are retrieved guided by these plans
3. Inference is executed over retrieved evidence

**Results**: 100% accuracy on MetaQA, outperforms best baselines by 1-5% on PathQuestion, WebQuestionsSP, ComplexWebQuestions.

#### PGDA-KGQA: Prompt-Guided Data Augmentation

**Paper**: arXiv:2506.09414

**Approach**: Enriches training sets through:
1. Generating single-hop pseudo questions
2. Semantic-preserving question rewriting
3. Answer-guided reverse path exploration for multi-hop questions

**Results**: +2.8% F1 on WebQSP, +1.8% F1 on ComplexWebQuestions

#### Tree-Based Reasoning with MCTS

Formulates KGQA as discrete decision-making, using Monte Carlo Tree Search to iteratively refine reasoning paths through the KG.

#### KG-Extended RAG

**Paper**: "Knowledge graph-extended retrieval augmented generation for question answering" (Applied Intelligence, 2025)

Combines traditional RAG with KG traversal for enriched context retrieval.

### Applicability to Multi-Agent KG Pipeline

- The **Plan-Retrieve-Reason** paradigm from DRKG maps well to multi-agent QA: one agent plans the query path, another retrieves from KG, a third reasons over evidence
- **Multi-hop reasoning** is critical for complex QA over KGs -- MCTS provides a principled exploration strategy
- **Data augmentation** strategies (pseudo question generation, reverse path exploration) could improve QA agent training
- **KG-extended RAG** shows the value of combining vector and graph retrieval (similar to LightRAG's mix mode)

---

## 10. Structured Extraction with Constrained Decoding

### Outlines (dottxt-ai)

**GitHub**: https://github.com/dottxt-ai/outlines (~13.6k stars)
**Docs**: https://dottxt-ai.github.io/outlines/

#### What It Does

Outlines guarantees structured outputs during LLM generation by constraining token selection at each step using finite-state machines compiled from JSON schemas, regex patterns, or context-free grammars.

#### How It Works

1. Compiles JSON Schema / regex / grammar into a **finite-state machine (FSM)**
2. Precomputes vocabulary index: maps each FSM state to valid next tokens
3. At each generation step, masks out invalid tokens (O(1) lookup per step)
4. Result: **guaranteed valid output** matching the schema

#### Supported Output Types

- JSON / Pydantic models
- Regular expressions
- Context-free grammars
- Literal values and enums
- Union types
- Complex nested structures

#### Backend Support

- OpenAI, Ollama, vLLM, HuggingFace Transformers
- Same code works across all backends

#### Performance

- **98% reduction in runtime overhead** with FSM-token-mask tensors
- XGrammar (related library): <40 microseconds per token mask generation

---

### Instructor (567-labs)

**GitHub**: https://github.com/567-labs/instructor (~12.6k stars, 3M+ monthly downloads)
**Docs**: https://python.useinstructor.com/

#### What It Does

Instructor uses Pydantic models to define structured extraction schemas, then handles LLM calls with automatic validation and retries.

#### How It Works

1. Define Pydantic model for desired output
2. Instructor converts to function call / structured output schema
3. LLM generates response
4. Pydantic validates output
5. If validation fails, **automatic retry with error feedback** to the LLM
6. **Semantic validation**: LLM-based validation of content correctness

#### Key Features

- **Automatic retries with error context**: Failed validations retry, passing the error back to the LLM
- **Streaming**: Partial objects stream as generated
- **Nested structures**: Handles complex hierarchical data
- **15+ LLM providers**: OpenAI, Anthropic, Google, Ollama, Groq, DeepSeek, etc.
- **Semantic validation**: Uses LLM to validate extracted content against natural language criteria

---

### BAML (BoundaryML)

**GitHub**: https://github.com/BoundaryML/baml (~7.8k stars)
**Docs**: https://docs.boundaryml.com/

#### What It Does

BAML is a domain-specific language (DSL) for building reliable AI workflows with structured outputs. It converts prompt engineering into "schema engineering."

#### How It Works

- Functions are defined in `.baml` files with explicit input variables and return types
- **Schema-Aligned Parsing (SAP)**: Proprietary algorithm that handles flexible LLM outputs (markdown within JSON, chain-of-thought before answering, etc.)
- Generates type-safe clients for Python, TypeScript, Ruby, Go, Java, C#, Rust
- Day-1 compatibility with new model releases

#### Key Features

- **100+ model support**: OpenAI, Anthropic, Google, AWS Bedrock, Ollama, vLLM, etc.
- **Retry policies and fallback mechanisms**: Statically defined
- **VSCode extension**: Prompt visualization and API request transparency
- **Offline-capable**: No internet dependency
- **Apache 2.0 license**

---

### vLLM Structured Decoding

**Docs**: https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html
**Blog**: https://blog.vllm.ai/2025/01/14/struct-decode-intro.html

vLLM natively supports structured generation using Outlines, lm-format-enforcer, or XGrammar as backends. Near-zero overhead constrained decoding at production scale.

---

### Applicability to Multi-Agent KG Pipeline

**This category is critical for solving the JSON parsing failures pain point.**

- **Outlines + vLLM**: If serving local models via vLLM, enable constrained decoding to guarantee valid JSON output for every extraction call. Eliminates parsing failures entirely.
- **Instructor**: Best choice for Ollama-based workflows. Define Pydantic models for entities/relations, get automatic validation and retries. The semantic validation feature could validate extraction quality.
- **BAML**: Best choice if building a polyglot system. SAP handles messy LLM outputs gracefully. The retry/fallback mechanisms add resilience.

**Recommended approach**: Define Pydantic schemas for KG extraction output (Entity, Relationship, Triple) and use Instructor or BAML to enforce them. This directly addresses:
- JSON parsing failures (guaranteed valid output)
- Wrong node connections (schema constraints + validation)
- Missed connections (retry logic + error feedback)

---

## 11. Supplementary: iText2KG and Production Best Practices

### iText2KG

**GitHub**: https://github.com/AuvaLab/itext2kg
**Paper**: "iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models" (WISE 2024)

**Architecture**: 4 modules:
1. **Document Distiller**: Reformulates raw documents into semantic blocks
2. **Incremental Entity Extractor**: Identifies unique entities, resolves ambiguities using cosine similarity against global entity set
3. **Incremental Relation Extractor**: Detects semantically unique relationships
4. **Graph Integrator**: Merges into unified KG

**Key Innovation**: Entity matching via cosine similarity with threshold -- local entities compared against global set to prevent duplicates while keeping distinct entities separate.

**Zero-shot, plug-and-play, no post-processing required.**

### Production Best Practices (from 2025 survey)

#### Entity Resolution Strategy (Multi-Step)

1. **Fuzzy matching**: Case-insensitive pattern matching for obvious duplicates
2. **Embedding similarity**: Vector representations with >0.95 similarity threshold
3. **Automated merging**: Consolidate properties and transfer relationships

#### Schema Design

- Start with **3-7 node types** and **5-15 relationship types**
- Involve domain experts upfront
- "80/20 rule: 20% of relationship types capture 80% of important connections"

#### Chunking Strategy

- **Overlapping chunks** (10% overlap recommended)
- Recursive character splitting at paragraph, sentence, and word levels
- Prevents relationship fragmentation across chunk boundaries

#### Model Selection Decision Matrix

| Scale | Approach | Accuracy | Hallucination Rate |
|-------|----------|----------|--------------------|
| <1,500 docs | Prompt-based (GPT-4o) | 70-80% | 12-25% |
| 1,500-10,000 docs | Fine-tune with QLoRA | +210% gain | 6.4% |
| >10,000 docs/month | Self-host Llama 3.1 70B | High | Low |

Break-even for fine-tuning: ~1,500 documents processed.

#### Quality Assurance Metrics

- Entities extracted per document
- Relationship density
- Orphaned node percentage
- Confidence score distribution
- Hallucination rate (vs. ground truth)

#### Entity Deduplication Best Practices

- **Normalize text to lowercase**
- **Resolve pronouns** before extraction
- **Iterative LLM-guided clustering** to merge equivalent entities beyond surface matching (KGGEN approach)
- **Strict merge criteria**: Only fully equivalent entities; exclude parent-child or technically distinct terms
- **ATOM framework**: Use cosine similarity for entity/relation resolution (avoids LLM-based resolution bottleneck)

---

## 12. Synthesis: Applicability to Multi-Agent KG Pipeline

### Mapping Research to Pipeline Pain Points

| Pain Point | Solutions from Research |
|-----------|----------------------|
| **Missed connections** | GraphRAG gleaning (iterative extraction), GLiNER+GLiREL as complementary first-pass, KARMA's multi-agent redundancy |
| **Wrong nodes connected** | Schema-constrained extraction (LLMGraphTransformer tuples, GLiREL entity type constraints, Triplex predicates), KARMA's conflict resolution agents, NuExtract's verbatim-string mode |
| **JSON parsing failures** | Outlines constrained decoding, Instructor validation+retries, BAML SAP algorithm, vLLM structured outputs |

### Recommended Architecture Patterns

**Tier 1: Fast Extraction (No LLM Cost)**
- GLiNER for entity extraction + GLiREL for relation classification
- REBEL/mREBEL for triplet generation (limited to Wikidata types)
- Runs on CPU, processes at 47+ sentences/second

**Tier 2: LLM-Based Extraction with Guarantees**
- Triplex via Ollama for schema-guided triplet extraction (3.8B params)
- NuExtract for template-driven structured extraction (2B-8B params)
- Constrained decoding via Outlines/Instructor/BAML for guaranteed valid output

**Tier 3: LLM-Based Verification and Enrichment**
- Larger LLM (32B+) for verifying extractions from Tier 1/2
- LLMGraphTransformer-style schema-constrained extraction with coreference resolution
- KARMA-style evaluator agent for quality gating

**Tier 4: Graph Construction and Querying**
- LightRAG for lightweight KG storage and dual-level retrieval
- iText2KG-style incremental entity resolution with cosine similarity
- Community detection (Leiden algorithm) for global query support

### Key Takeaways

1. **Use constrained decoding immediately** -- Instructor or Outlines will eliminate JSON parsing failures with minimal code changes
2. **GLiNER+GLiREL as a complementary extraction agent** -- zero-cost, fast, and catches patterns LLMs miss
3. **KARMA's 9-agent architecture** validates the multi-agent approach but shows that schema alignment and conflict resolution deserve their own dedicated agents
4. **Triplex on Ollama** provides an immediately deployable, schema-guided extraction model that outperforms GPT-4 at KG tasks
5. **NuExtract's verbatim-string mode** is ideal for entity name extraction (zero hallucination on names)
6. **LightRAG's dual-level retrieval** provides the best cost-performance ratio for KG-based QA
7. **Entity resolution is the critical unsolved challenge** -- iText2KG's cosine similarity approach and production best practices (fuzzy matching + embedding similarity + automated merging) provide practical solutions
8. **Schema design matters more than model selection** -- start with 3-7 node types, 5-15 relationship types, and iterate
