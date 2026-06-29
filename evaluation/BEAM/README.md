# BEAM Benchmark Evaluation

**BEAM** — *Beyond a Million Tokens* (ICLR 2026)  
Paper: https://arxiv.org/abs/2510.27246  
Dataset: https://huggingface.co/datasets/Mohammadta/BEAM

100 conversations (128K–10M tokens), 2,000 questions, 10 memory ability types.

---

## Two-Phase Protocol

**Phase 1 — Answer Generation (`run_eval.py`)**
- Downloads BEAM from HuggingFace
- Builds KG from each conversation via `DeliberativeOrchestrator`
- Answers every probing question via `AdvancedQAOrchestrator`
- Saves raw responses to JSON (incremental — safe to interrupt)

**Phase 2 — Scoring (`score.py`)**
- Loads responses JSON
- Runs LLM-as-judge on each answer against its rubric
- Outputs per-type scores + overall aggregate
- Resumable — use `--resume` to skip already-scored entries

---

## Prerequisites

```bash
# Required for HuggingFace dataset download
pip install datasets

# Required for event_ordering Kendall-tau metric (optional but recommended)
pip install sentence-transformers scipy

# Required for robust JSON repair in judge output
pip install json-repair
```

---

## Recommended Model Config

Set in `.env` before running:

```bash
# Backbone — KG extraction + QA answering
LLM_BACKEND=vllm
VLLM_BASE_URL=https://<your-endpoint>/v1
VLLM_API_KEY=<your-key>
LLM_DEFAULT_MODEL=deepseek-ai/DeepSeek-V4-Flash

# Embedding (for vector retrieval)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Judge — can be a different model/endpoint
# Set BEAM_JUDGE_MODEL to override LLM_DEFAULT_MODEL for scoring only
BEAM_JUDGE_MODEL=Qwen/Qwen3.7-Plus
```

---

## Quick Start — 100K Tier

### Smoke test (3 chats, 5 questions each)
```bash
python evaluation/BEAM/run_eval.py \
    --tier 100K \
    --max-samples 3 \
    --max-questions 5 \
    --output evaluation/results/beam_100K_smoke_responses.json
```

### Score the smoke test
```bash
python evaluation/BEAM/score.py \
    --responses evaluation/results/beam_100K_smoke_responses.json \
    --judge-model Qwen/Qwen3.7-Plus \
    --output evaluation/results/beam_100K_smoke_scores.json
```

---

## Full 100K Run

### Phase 1 — Build KGs + answer all questions
```bash
python evaluation/BEAM/run_eval.py \
    --tier 100K \
    --chunk-size 3000 \
    --retrieval hybrid \
    --save-kg-dir evaluation/results/beam_kg_cache/100K \
    --output evaluation/results/beam_100K_responses.json
```

### Phase 1 — Re-run QA only (KGs already built)
```bash
python evaluation/BEAM/run_eval.py \
    --tier 100K \
    --load-kg-dir evaluation/results/beam_kg_cache/100K \
    --output evaluation/results/beam_100K_responses.json
```

### Phase 2 — Score
```bash
python evaluation/BEAM/score.py \
    --responses evaluation/results/beam_100K_responses.json \
    --judge-model Qwen/Qwen3.7-Plus \
    --output evaluation/results/beam_100K_scores.json
```

### Phase 2 — Resume interrupted scoring
```bash
python evaluation/BEAM/score.py \
    --responses evaluation/results/beam_100K_responses.json \
    --judge-model Qwen/Qwen3.7-Plus \
    --output evaluation/results/beam_100K_scores.json \
    --resume
```

---

## Tiers

| Tier | Chats | Questions (est.) | Backbone cost | Judge cost |
|------|-------|-----------------|---------------|------------|
| **100K** | 20 | ~400 | ~$3–5 | ~$1.50 |
| **500K** | 35 | ~700 | ~$15–25 | ~$2.50 |
| **1M** | 35 | ~700 | ~$30–50 | ~$2.50 |
| **10M** | 10 | ~1,150/chat | not feasible locally | – |

*Costs estimated with DeepSeek-V4-Flash backbone + Qwen3.7-Plus judge.*

---

## Output Format

### Responses file (`run_eval.py` output)
```json
{
  "meta": {"model": "...", "n_chats": 20, "n_questions": 412},
  "chats": [
    {
      "conversation_id": "1",
      "domain": "Coding",
      "responses": {
        "abstention": [
          {"question": "...", "rubric": ["..."], "llm_response": "..."}
        ],
        "information_extraction": [...],
        ...
      }
    }
  ]
}
```

### Scores file (`score.py` output)
```json
{
  "meta": {"model": "...", "judge_model": "Qwen/Qwen3.7-Plus", ...},
  "aggregate": {
    "overall": 0.412,
    "n_questions": 412,
    "per_type": {
      "abstention": 0.55,
      "contradiction_resolution": 0.38,
      "event_ordering": 0.41,
      "information_extraction": 0.62,
      "instruction_following": 0.71,
      "knowledge_update": 0.58,
      "multi_session_reasoning": 0.49,
      "preference_following": 0.73,
      "summarization": 0.55,
      "temporal_reasoning": 0.44
    }
  },
  "chats": [...]
}
```

---

## Comparison Targets (published baselines, 100K / 1M tier)

| System | Backbone | 100K overall | 1M overall |
|--------|----------|-------------|-----------|
| LIGHT  | Llama 4 Maverick | ~0.358 | ~0.336 |
| LIGHT  | GPT-4.1-nano | ~0.345 | ~0.336 |
| LIGHT  | Qwen 2.5-32B | ~0.311 | ~0.309 |
| RAG    | Llama 4 Maverick | ~0.323 | ~0.307 |
| Mem0   | undisclosed | – | 0.641 |

*Beat LIGHT + Qwen 2.5 (0.311) to be competitive. Beat LIGHT + Llama 4 (0.358) to be strong.*

**Note:** Comparability depends on judge model. BEAM paper used GPT-4o-class judge.
Using Qwen3.7-Plus as judge may shift scores ±5% vs published numbers.
Document your judge model clearly when reporting results.

---

## Question Types

| Type | Abbrev | Tests |
|------|--------|-------|
| abstention | ABS | Refuse when info was never mentioned |
| contradiction_resolution | CR | Reconcile conflicting statements |
| event_ordering | EO | Reconstruct sequence of events |
| information_extraction | IE | Recall entities/facts from deep history |
| instruction_following | IF | Sustained adherence to old constraints |
| knowledge_update | KU | Track when facts change |
| multi_session_reasoning | MR | Connect evidence across non-adjacent turns |
| preference_following | PF | Adapt to evolving user preferences |
| summarization | SUM | Compress/abstract the conversation |
| temporal_reasoning | TR | Reason about time relations |
