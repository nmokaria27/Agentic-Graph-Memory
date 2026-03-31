# Evaluation Framework for Multi-Agent KG Extraction

This directory contains tools for evaluating the multi-agent knowledge graph
extraction pipeline against gold-standard benchmarks.

## Directory Structure

```
evaluation/
  evaluate_kg.py           # Core evaluation metrics (P/R/F1, hallucination, etc.)
  run_evaluation.py        # End-to-end runner: load data, run pipeline, evaluate
  README.md                # This file
  adapters/
    scierc_adapter.py      # Converts SciERC data to pipeline format
  datasets/
    scierc/                # SciERC benchmark data (350 train / 50 dev / 100 test)
  results/                 # Per-document pipeline outputs (created at runtime)
  kgafe/                   # KG-Grounded Atomic Fact Evaluation framework
    evaluator.py           # Main KGAFE evaluator (orchestrates the pipeline)
    atomic_decomposer.py   # Decomposes answers into atomic facts
    triple_verifier.py     # 3-tier verification (exact, path-based, semantic)
    judge_panel.py         # Multi-judge evaluation (correctness, completeness, groundedness)
    benchmark_generator.py # Auto-generates QA benchmarks from KG structure
    run_kgafe.py           # CLI runner for KGAFE evaluation
```

## Available Datasets

### SciERC

The SciERC dataset (Luan et al., 2018) contains 500 scientific abstracts from
AI conference/workshop proceedings, annotated with:

- **6 entity types:** Task, Method, Metric, Material, OtherScientificTerm, Generic
- **7 relation types:** Used-for, Feature-of, Part-of, Compare, Hyponym-of,
  Conjunction, Evaluate-for

Split sizes: 350 train / 50 dev / 100 test documents.

Source: http://nlp.cs.washington.edu/sciIE/

## Quick Start

### 1. Evaluate pre-computed results

If you have already run the pipeline and saved per-document JSON files in
`evaluation/results/`:

```bash
python evaluation/run_evaluation.py --skip-pipeline --results-dir evaluation/results/
```

### 2. Run the full pipeline and evaluate

```bash
# Run on 5 test documents (for quick testing):
python evaluation/run_evaluation.py --max-docs 5

# Full test set:
python evaluation/run_evaluation.py
```

### 3. Evaluate a single kg_export.json against a single gold document

```bash
python evaluation/evaluate_kg.py \
    --gold evaluation/datasets/scierc/test.json \
    --predicted kg_export.json
```

### 4. Convert SciERC data for inspection

```bash
python evaluation/adapters/scierc_adapter.py \
    --input evaluation/datasets/scierc/test.json \
    --output-pipeline pipeline_input.json \
    --output-gold gold_standard.json \
    --max-docs 5
```

## Metrics

The evaluation framework computes:

| Metric | Description |
|--------|-------------|
| **Entity P/R/F1 (strict)** | Exact normalised text match |
| **Entity P/R/F1 (partial)** | Token-overlap Jaccard >= 0.5 |
| **Entity type accuracy** | Of matched entities, fraction with correct type |
| **Entity hallucination rate** | Fraction of predicted entities with no gold match |
| **Triple P/R/F1 (strict)** | Exact match on (subject, relation, object) |
| **Triple P/R/F1 (fuzzy)** | Fuzzy subject/object match, exact relation |
| **Triple hallucination rate** | Fraction of predicted triples with no gold match |
| **Per-type breakdowns** | P/R/F1 broken down by entity type and relation type |

## Command-Line Options

### evaluate_kg.py

| Flag | Description |
|------|-------------|
| `--gold PATH` | Path to gold standard (SciERC JSON-lines or converted array) |
| `--predicted PATH` | Path to predicted kg_export.json or directory of per-doc JSONs |
| `--fuzzy-threshold F` | Similarity threshold for fuzzy matching (default: 0.8) |
| `--max-docs N` | Limit evaluation to first N gold documents |
| `--output-json PATH` | Write metrics to JSON file |

### run_evaluation.py

| Flag | Description |
|------|-------------|
| `--split {train,dev,test}` | SciERC split (default: test) |
| `--max-docs N` | Limit to N documents |
| `--skip-pipeline` | Only evaluate pre-computed results |
| `--results-dir PATH` | Directory for per-document results |
| `--model NAME` | LLM model to use (default: gemma3:27b) |
| `--fuzzy-threshold F` | Fuzzy matching threshold (default: 0.8) |
| `--include-generic` | Include Generic entities in evaluation |
| `--output-json PATH` | Save metrics as JSON |

## Pipeline Output Format

The pipeline produces `kg_export.json` with this structure:

```json
{
  "knowledge_graph": {
    "entities": [
      {"id": "entity_name", "type": "Method", "labels": [...], "metadata": {...}}
    ],
    "triples": [
      {
        "subject": "entity_a",
        "relation": "Used-for",
        "object": "entity_b",
        "confidence": 0.85
      }
    ]
  }
}
```

---

## KGAFE: KG-Grounded Atomic Fact Evaluation

KGAFE evaluates QA answers by decomposing them into atomic facts and verifying
each one against the knowledge graph. It combines structural graph verification
with LLM-based judgment.

### How it works

1. **Atomic Decomposition** -- Break the answer into independently verifiable
   facts (e.g., "HOMA-IR is a marker of insulin resistance").

2. **Three-Tier Verification** -- Each fact is checked in a cascade:
   - **Tier 1 (Exact Match)**: Direct KG triple lookup. No LLM needed.
   - **Tier 2 (Path-Based)**: Multi-hop paths (up to 3 hops) between entities,
     then LLM checks if the path logically entails the fact.
   - **Tier 3 (Semantic)**: 2-hop entity neighbourhood + LLM semantic judgment.

3. **Judge Panel** -- Three independent LLM judges score the answer:
   - Correctness (35% weight): Are the facts accurate?
   - Completeness (25% weight): Does the answer cover the question?
   - Groundedness (40% weight): Are claims traceable to KG triples?

4. **KGAFE Score** -- Weighted composite:
   ```
   KGAFE = 0.30 * KG_Faithfulness + 0.25 * (1 - Hallucination_Rate)
         + 0.20 * Groundedness + 0.15 * Coverage + 0.10 * Correctness
   ```

### Quick start

```bash
# Evaluate a single question/answer pair
python -m evaluation.kgafe.run_kgafe \
  --kg-path kg_export.json \
  --question "What is HOMA-IR?" \
  --answer "HOMA-IR is a marker of insulin resistance."

# Skip judge panel for faster evaluation (verification only)
python -m evaluation.kgafe.run_kgafe \
  --kg-path kg_export.json \
  --question "What is HOMA-IR?" \
  --answer "HOMA-IR is a marker of insulin resistance." \
  --no-judge

# Run full auto-benchmark (generates questions from KG, evaluates QA system)
python -m evaluation.kgafe.run_kgafe \
  --kg-path kg_export.json --benchmark --n-questions 20

# Save results to file
python -m evaluation.kgafe.run_kgafe \
  --kg-path kg_export.json --benchmark --output results/kgafe_benchmark.json
```

### KGAFE metrics

| Metric | Description |
|--------|-------------|
| **KG Faithfulness** | Fraction of atomic facts supported by the KG |
| **Hallucination Rate** | Fraction of facts that are contradicted or unverifiable |
| **Groundedness** | Judge panel score for how traceable claims are to KG triples |
| **Coverage** | How much of the relevant KG content the answer mentions |
| **Correctness** | Judge panel score for factual accuracy |
| **Completeness** | Judge panel score for question coverage |
| **KGAFE Score** | Weighted composite of all metrics (0-1) |
| **Tier Distribution** | How many facts verified at each tier (exact/path/semantic) |
| **Verdict Distribution** | Counts of supported/contradicted/unverifiable facts |

### Sample results

See `kgafe_results_fixed.json` in the repo root for a full evaluation run
(4 questions, 97 atomic facts, 96 supported, 0 contradicted). See
`demo_results.json` for the complete QA pipeline output including expert
responses, debate transcripts, and provenance chains.

---

## References

- Luan, Y., He, L., Ostendorf, M., & Hajishirzi, H. (2018). Multi-Task
  Identification of Entities, Relations, and Coreference for Scientific
  Knowledge Graph Construction. In EMNLP 2018.
