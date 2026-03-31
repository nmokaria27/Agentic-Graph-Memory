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
    scierc/
      train.json           # SciERC training set (350 abstracts)
      dev.json             # SciERC development set (50 abstracts)
      test.json            # SciERC test set (100 abstracts)
      raw_data/            # Raw SciERC text files and annotations
      processed_data/      # Original processed SciERC directory
  results/                 # Per-document pipeline outputs (created at runtime)
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

## References

- Luan, Y., He, L., Ostendorf, M., & Hajishirzi, H. (2018). Multi-Task
  Identification of Entities, Relations, and Coreference for Scientific
  Knowledge Graph Construction. In EMNLP 2018.
