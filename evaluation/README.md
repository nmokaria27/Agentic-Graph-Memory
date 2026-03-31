# Evaluation Framework

Two evaluation frameworks: **SciERC** for benchmarking KG extraction quality,
and **KGAFE** for evaluating QA answer faithfulness against the KG.

## Directory Structure

```
evaluation/
  evaluate_kg.py           # KG extraction metrics (P/R/F1, hallucination, etc.)
  run_evaluation.py        # SciERC end-to-end runner
  adapters/
    scierc_adapter.py      # Converts SciERC data to pipeline format
  datasets/
    scierc/                # SciERC benchmark (350 train / 50 dev / 100 test)
  results/                 # Per-document outputs (created at runtime)
  kgafe/                   # KG-Grounded Atomic Fact Evaluation
    evaluator.py           # Main KGAFE evaluator (orchestrates the pipeline)
    atomic_decomposer.py   # Decomposes answers into atomic facts
    triple_verifier.py     # 3-tier verification (exact, path-based, semantic)
    judge_panel.py         # Multi-judge panel (correctness, completeness, groundedness)
    benchmark_generator.py # Auto-generates QA benchmarks from KG structure
    run_kgafe.py           # CLI runner for KGAFE evaluation
```

---

## SciERC Benchmark

Evaluates KG extraction against the SciERC dataset (Luan et al., 2018) --
500 scientific abstracts annotated with 6 entity types and 7 relation types.

```bash
# Quick test (5 documents)
python evaluation/run_evaluation.py --max-docs 5

# Full test set
python evaluation/run_evaluation.py

# Evaluate pre-computed results
python evaluation/run_evaluation.py --skip-pipeline --results-dir evaluation/results/
```

Metrics: Entity P/R/F1 (strict + partial), type accuracy, triple P/R/F1
(strict + fuzzy), hallucination rates, per-type breakdowns.

---

## KGAFE: KG-Grounded Atomic Fact Evaluation

KGAFE evaluates QA answers by decomposing them into atomic facts and verifying
each one against the knowledge graph. It combines structural graph verification
with LLM-based judgment to produce a comprehensive faithfulness score.

### Pipeline

```
Answer --> [Atomic Decomposer] --> [Three-Tier Verifier] --> [Judge Panel] --> KGAFE Score
```

**1. Atomic Decomposition**

Breaks the answer into independently verifiable facts following the FActScore
principle. Each fact is a single claim mentioning at least one KG entity.

Example: "HOMA-IR, a marker of insulin resistance, mediates microvascular dysfunction" becomes:
- "HOMA-IR is a marker of insulin resistance" (definition)
- "HOMA-IR mediates microvascular dysfunction" (claim)

**2. Three-Tier Verification**

Each fact is checked in a cascade -- the most precise method first, falling back
to broader ones:

| Tier | Method | How it works | LLM needed? |
|------|--------|-------------|-------------|
| **Tier 1** | Exact Match | Direct KG triple lookup between resolved entities | No |
| **Tier 2** | Path-Based | Find multi-hop paths (up to 3 hops), LLM checks if the path entails the fact | Yes |
| **Tier 3** | Semantic | Gather 2-hop entity neighbourhood, LLM judges semantic entailment | Yes |

Each fact gets a verdict: **supported**, **contradicted**, **partially supported**, or **unverifiable**.

Entity resolution uses aggressive normalization (`normalize_for_matching()`) to
bridge entity IDs (`homair`), display names (`HOMA-IR`), and snake_case (`homa_ir`).

**3. Judge Panel**

Three independent LLM judges evaluate the full answer:

| Judge | Weight | Question |
|-------|--------|----------|
| Correctness | 35% | Are the facts accurate compared to KG evidence? |
| Completeness | 25% | Does the answer cover all relevant aspects? |
| Groundedness | 40% | Is every claim traceable to KG triples? |

A meta-judge aggregates the weighted scores. Overall pass requires all three
judges to pass their binary thresholds AND a weighted score >= 0.65.

**4. KGAFE Score**

```
KGAFE = 0.30 * KG_Faithfulness + 0.25 * (1 - Hallucination_Rate)
      + 0.20 * Groundedness + 0.15 * Coverage + 0.10 * Correctness
```

Weighting prioritizes faithfulness and anti-hallucination (55% combined),
then groundedness (20%), coverage (15%), and correctness (10%).

### Usage

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

### Metrics

| Metric | Description |
|--------|-------------|
| **KG Faithfulness** | Fraction of atomic facts supported by the KG |
| **Hallucination Rate** | Fraction of facts contradicted or unverifiable |
| **Groundedness** | Judge score for how traceable claims are to KG triples |
| **Coverage** | How much of the relevant KG content the answer mentions |
| **Correctness** | Judge score for factual accuracy |
| **Completeness** | Judge score for question coverage |
| **KGAFE Score** | Weighted composite of all metrics (0-1) |
| **Tier Distribution** | How many facts verified at each tier (exact/path/semantic) |
| **Verdict Distribution** | Counts of supported/contradicted/unverifiable facts |

### Auto-Benchmark Generator

KGAFE can auto-generate QA benchmarks directly from KG structure with provably
correct gold answers (no human annotation needed):

- **Single-hop**: Direct triple-based Q&A
- **Multi-hop**: Path-based reasoning (2-3 hop chains)
- **Aggregation**: "What are all the things that X relates to?"
- **Comparison**: "Compare X and Y"
- **Negative**: "Does X relate to Z?" (when it doesn't)

### Sample Results

See `kgafe_results_fixed.json` in the repo root for a full evaluation run
(4 questions, 97 atomic facts, 96 supported, 0 contradicted, avg KGAFE 0.90).
See `demo_results.json` for the complete QA pipeline output including expert
responses, debate transcripts, critic reviews, and provenance chains.

---

## References

- Luan, Y., He, L., Ostendorf, M., & Hajishirzi, H. (2018). Multi-Task
  Identification of Entities, Relations, and Coreference for Scientific
  Knowledge Graph Construction. In EMNLP 2018.
