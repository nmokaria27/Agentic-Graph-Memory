# Benchmark Evaluation Guide — LoComo & MemoryAgentBench

## 1. LoComo (Long-term Conversational Memory)

### What it is
LoComo ([SNAP Research, 2024](https://github.com/snap-research/locomo)) is a benchmark for testing **long-term conversational memory**. It contains **10 conversations** spanning months of sessions between two people, with 300+ QA pairs total. Each conversation includes multiple sessions with dates, shared images, and natural back-and-forth dialogue.

### What it tests
It evaluates whether your system can retain, organize, and retrieve information from **extended multi-session conversations** — the kind that accumulate personal facts, events, opinions, and temporal details over time.

**5 question categories:**

| Cat | Name | Tests | Example |
|-----|------|-------|---------|
| 1 | **Multi-hop** | Connecting facts across sessions | "What are all the places X recommended?" |
| 2 | **Temporal** | Date/time reasoning from session timestamps | "When did Y start their new job?" |
| 3 | **Open-domain** | General recall with multiple valid answers | "What hobbies does X have?" |
| 4 | **Single-hop** | Direct fact retrieval | "What is Y's dog's name?" |
| 5 | **Adversarial** | Correctly abstaining when info was never mentioned | "What is X's favorite movie?" (never discussed) |

### Metrics
- **Token F1** — overlap between predicted and gold answer tokens (primary for cats 1-4)
- **Abstention accuracy** — for adversarial questions (cat 5), must say "no information available"
- **Per-category breakdown** — lets you see where your system is strong/weak

### How your adapter works
`evaluation/LoComo/run_eval.py` does:
1. Loads `locomo10.json` (you already have it at `evaluation/LoComo/data/locomo10.json` — 2.8MB)
2. Formats each conversation into flat text
3. Builds a GovernedKnowledgeGraph via `DeliberativeOrchestrator`
4. Constructs `AdvancedQAOrchestrator` with domain experts + vector retrieval
5. Queries each question, scores F1 per the LoComo paper spec

### How to run

```bash
# Smoke test — 2 conversations, 10 Qs each (~15–30 min with Gemma 31B)
python evaluation/LoComo/run_eval.py \
    --data-file evaluation/LoComo/data/locomo10.json \
    --max-samples 2 --max-questions 10 \
    --output evaluation/results/locomo_smoke.json

# Save KGs to skip pipeline on re-run
python evaluation/LoComo/run_eval.py \
    --data-file evaluation/LoComo/data/locomo10.json \
    --max-samples 2 --max-questions 10 \
    --save-kg-dir evaluation/results/locomo_kg_cache \
    --output evaluation/results/locomo_smoke.json

# Re-run QA only (load cached KGs)
python evaluation/LoComo/run_eval.py \
    --data-file evaluation/LoComo/data/locomo10.json \
    --load-kg-dir evaluation/results/locomo_kg_cache \
    --output evaluation/results/locomo_full.json

# Full run — all 10 conversations, all questions (hours-long)
python evaluation/LoComo/run_eval.py \
    --data-file evaluation/LoComo/data/locomo10.json \
    --output evaluation/results/locomo_full.json
```

---

## 2. MemoryAgentBench

### What it is
MemoryAgentBench ([ai-hyz, 2024](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)) is a benchmark that tests **memory-augmented LLM agents** across three distinct capabilities. It downloads datasets from HuggingFace automatically — no cloning needed.

### What it tests

| Dataset | Category | What it measures |
|---------|----------|-----------------|
| **eventqa** | Accurate Retrieval | Can you find specific facts from large text contexts? |
| **ruler_qa** | Accurate Retrieval | Needle-in-a-haystack retrieval from very long documents |
| **detectiveqa** | Long-Range Understanding | Can you reason across widely separated passages? |
| **fact_mh** | Conflict Resolution | Multi-hop: when facts conflict/update, which is correct? |
| **fact_sh** | Conflict Resolution | Single-hop: handling contradictory information |

**Three core capabilities:**
- **Accurate Retrieval** — Precise fact lookup from large contexts. Tests whether your KG actually captures and retrieves the right information.
- **Long-Range Understanding** — Reasoning that requires connecting evidence scattered across a long document. Your multi-hop graph traversal is directly tested here.
- **Conflict Resolution** — When information changes or contradicts itself, does your system return the most current/correct version? This tests your governance layer's ability to handle updates.

### Metrics
- **Substring Exact Match (SEM)** — primary metric for Accurate Retrieval & Conflict Resolution: gold answer is a substring of prediction
- **Exact Match (EM)** — primary for Long-Range Understanding
- **Token F1** — secondary metric across all datasets
- **Memory build time** / **Query time** — efficiency tracking

### How your adapter works
`evaluation/Memory-Agent-Bench/agent_graph_memory_adapter.py` wraps your system as an `AgentGraphMemoryWrapper`:
1. **Memorize phase**: text chunks are accumulated, KG is built lazily on first query
2. **Query phase**: `AdvancedQAOrchestrator` answers over the KG with hybrid retrieval

`run_eval.py` orchestrates: download from HF → chunk context → memorize → query → score.

### How to run

```bash
# Prerequisite: install HuggingFace datasets
pip install datasets

# Smoke test — eventqa, 5 examples
python evaluation/Memory-Agent-Bench/run_eval.py \
    --dataset eventqa --max-samples 5 \
    --output evaluation/results/mab_eventqa_smoke.json

# With KG caching for re-runs
python evaluation/Memory-Agent-Bench/run_eval.py \
    --dataset eventqa --max-samples 5 \
    --save-kg-dir evaluation/results/mab_kg_cache \
    --output evaluation/results/mab_eventqa_smoke.json

# Full eventqa run
python evaluation/Memory-Agent-Bench/run_eval.py \
    --dataset eventqa --max-samples -1 \
    --output evaluation/results/mab_eventqa_full.json

# Conflict resolution test
python evaluation/Memory-Agent-Bench/run_eval.py \
    --dataset fact_mh --max-samples 5 \
    --output evaluation/results/mab_fact_mh.json

# All 5 datasets
python evaluation/Memory-Agent-Bench/run_eval.py \
    --dataset all --max-samples 5 \
    --output evaluation/results/mab_all_smoke.json
```

---

## 3. Prerequisites Before Running

### Environment setup
Both benchmarks use your existing pipeline and LLM backend. Ensure your `.env` is configured:
- `LLM_BACKEND`, `VLLM_BASE_URL`, `LLM_DEFAULT_MODEL` — for LLM calls
- `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` — for vector embeddings
- SSH tunnels to gpu01/gpu02 must be active

### Dependencies
```bash
pip install -e .              # Install multi_agent_kg package
pip install datasets          # Required for MemoryAgentBench (HuggingFace download)
pip install nltk              # Required for LoComo (PorterStemmer for F1 scoring)
```

### Verify tunnels are up
```bash
lsof -iTCP:8000 -iTCP:11435 -sTCP:LISTEN
```

---

## 4. Recommended Testing Order

1. **LoComo smoke** (2 convos, 10 Qs) — fastest way to see if pipeline works end-to-end
2. **MemoryAgentBench eventqa** (5 examples) — tests accurate retrieval
3. **MemoryAgentBench fact_mh** (5 examples) — tests conflict resolution (unique to memory systems)
4. **MemoryAgentBench detectiveqa** (5 examples) — tests long-range understanding
5. Scale up `--max-samples -1` for full benchmark numbers

### What good scores look like (approximate targets)

| Benchmark | Metric | Baseline (RAG) | Good | Excellent |
|-----------|--------|-----------------|------|-----------|
| LoComo overall | F1 | ~0.25 | ~0.40 | ~0.55+ |
| LoComo adversarial | Abstention | ~0.30 | ~0.70 | ~0.90+ |
| MAB eventqa | SEM | ~0.40 | ~0.60 | ~0.80+ |
| MAB fact_mh | SEM | ~0.20 | ~0.40 | ~0.60+ |
| MAB detectiveqa | EM | ~0.15 | ~0.30 | ~0.50+ |

These are rough guideposts based on typical memory system performance; published baselines vary.
