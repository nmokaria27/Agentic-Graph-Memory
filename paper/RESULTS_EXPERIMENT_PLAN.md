# Governed KG Results Plan

## Goal

Show that the Governed Knowledge Graph is better than a flat KG in three measurable ways:

1. **Better admitted graph quality**
   - fewer bad triples
   - better triple F1
   - lower triple hallucination
2. **Better graph structure**
   - domain ownership
   - cross-domain organization
   - auditability
3. **Better downstream use**
   - better grounded QA than flat retrieval baselines such as RAG / GraphRAG

This is a **governed-memory paper**, not an extraction-SOTA paper. Extraction results must be good enough to show competence, but the main claim is that governance improves the resulting knowledge structure and the quality of what gets admitted.

## Main Paper Claims

### Claim A: Governance improves the admitted graph

Use governed (`triage`) vs flat (`unguided`) comparisons.

Main metrics:
- entity strict F1
- triple strict F1
- triple hallucination
- entity overlap
- triple overlap
- cross-domain fraction
- overhead

Interpretation:
- if entity F1 is similar but triple F1 is higher and triple hallucination is lower under governance, then governance improves the admitted relational graph rather than merely changing metadata

### Claim B: Governance changes graph structure

Use overlap + cross-domain metrics.

Main metrics:
- entity overlap Jaccard
- triple overlap Jaccard
- governed cross-domain fraction

Interpretation:
- low triple overlap means governance is changing the graph structure itself
- nontrivial cross-domain fraction means the governed graph is organizing knowledge across expert boundaries

### Claim C: Governance scales structurally

Use governed-only structure results at 100 docs.

Main metrics:
- routing exact match
- domain coverage
- governance completeness
- audit integrity

Interpretation:
- even if extraction is still noisy, the governed-memory structure remains correct and auditable at scale

### Claim D: Governance can support better downstream QA

Do this only after the governed-vs-flat KG comparison is stable.

Main metrics:
- answer correctness / exact match / token F1 where a gold answer exists
- faithfulness / groundedness
- attribution quality / evidence support
- domain-routing correctness
- abstention on unsupported questions

Interpretation:
- the governed KG should not only store knowledge differently, but make it easier to answer questions with grounded, domain-routed reasoning

## Results Section Structure

### 1. Extraction Competence

Purpose:
- show the pipeline is good enough to build a usable KG

Report:
- 10-doc governed vs flat
- 50-doc governed vs flat

Metrics:
- entity strict F1
- triple strict F1
- triple hallucination

### 2. Governance as an Intervention

Purpose:
- show governance improves the admitted relational graph

Report:
- governed vs flat at 10 docs
- governed vs flat at 50 docs

Metrics:
- entity strict F1
- triple strict F1
- triple hallucination
- entity overlap
- triple overlap
- cross-domain fraction
- overhead

### 3. Governance Structure at Scale

Purpose:
- show the governed memory remains correct at larger corpus scale

Report:
- governed 100-doc structure metrics

Metrics:
- routing exact match
- domain coverage
- governance completeness
- audit integrity
- cross-domain fraction
- orphan fraction
- zero-triple docs

### 4. Strict Governance Review (Supporting)

Purpose:
- show review can block bad facts, even if strict mode is too conservative to use as the main operating mode

Report:
- accept recall
- reject recall
- false accept rate

### 5. Downstream QA (Optional but Valuable)

Purpose:
- show the governed data structure is useful, not only elegant

Compare:
- flat RAG
- GraphRAG
- governed-KG QA

Metrics:
- answer correctness
- answer faithfulness
- evidence attribution
- domain routing correctness
- unsupported-question abstention

## Current Best Story

At present, the strongest story is:

- governed vs flat at 10 docs: large win on triple quality
- governed vs flat at 50 docs: governance keeps entity quality flat, improves triple F1, and reduces triple hallucination
- governed-only at 100 docs: routing / coverage / auditability remain near-perfect

This is already enough for a strong governed-memory paper.

## Why Extraction Is Still Weak

The main bottleneck is **relation extraction**, not governance.

Fixed schema helps by constraining relation labels, but it does not solve:
- relation proposal recall
- subject/object binding
- directionality
- exact-match alignment to benchmark entities

Current governance improves **precision** by filtering bad triples, but it does not recover missing triples.

The current extraction bottlenecks are:
- weak `Used-for` recall
- multi-stage relation binding loss (stage 1 type -> stage 2 head -> stage 3 tail)
- orphan entities and zero-triple docs
- prompt-based relation extraction rather than pairwise constrained classification

## Code Changes Most Likely to Improve Extraction

These are ranked by value within the current architecture.

### 1. Better relation prompts and pair selection

Goal:
- improve recall without hard-coding

Tasks:
- strengthen the `Used-for` instruction with multiple SciERC-style positive examples
- add explicit negative examples for common confusions (`Part-of`, `Feature-of`, `Conjunction`)
- limit candidate entity pairs using local heuristics:
  - same sentence
  - neighboring sentence
  - lexical trigger overlap
  - compatible types
- ask the model to score top-k candidate pairs rather than generate free triples from scratch

### 2. Add relation proposal diagnostics to the paper pipeline

Tasks:
- always record:
  - candidate pairs considered
  - relation types proposed
  - relation types surviving binding
  - triples rejected by governance
- use these logs to separate:
  - proposal failure
  - binding failure
  - governance rejection

### 3. Tune triage risk triggers

Goal:
- keep governance precision gains without over-filtering good triples

Tasks:
- inspect which reviewed triples end up rejected
- check whether `Part-of` or other relations are over-penalized by cross-domain review
- reduce unnecessary review on relation types that are already high precision

### 4. Add a benchmark-mode pairwise relation classifier

This is the highest-value architecture upgrade if time allows.

Idea:
- for fixed-schema SciERC experiments, replace or supplement the current generative relation-binding stack with:
  - candidate entity pair enumeration
  - constrained classification into the SciERC relation set or NONE

This is still compatible with the governed architecture and is the cleanest path to materially higher triple F1.

## Immediate Experiment Queue

### Priority 1: Lock the governed-vs-flat table

Need:
- 10-doc governed vs flat
- 50-doc governed vs flat

Metrics:
- entity strict F1
- triple strict F1
- triple hallucination
- entity overlap
- triple overlap
- cross-domain fraction
- overhead

### Priority 2: Freeze the 100-doc structure table

Need:
- governed-only 100-doc metrics

Metrics:
- routing exact match
- domain coverage
- governance completeness
- audit integrity
- cross-domain fraction
- orphan fraction
- zero-triple docs

### Priority 3: Run a small QA benchmark

Need:
- a compact SciERC-derived QA set or document-grounded QA set
- three systems:
  - flat RAG
  - GraphRAG
  - governed-KG QA

Metrics:
- answer correctness
- faithfulness
- attribution
- domain routing correctness

### Priority 4: Improve the relation extractor

Need:
- targeted `Used-for` improvement
- candidate-pair restriction
- stronger benchmark-mode prompting

## What Not To Do

- do not present sliced 5-doc or 10-doc subsets from a 50-doc accumulating run as clean standalone experiments
- do not center the paper on audit-only vs triage
- do not center the paper on strict governance review
- do not claim state-of-the-art extraction
- do not let enrichment become the main story unless a patched rerun is clean and stable

## Minimal Paper-Ready Package

If time gets tight, the minimum set of results needed is:

1. 10-doc governed vs flat
2. 50-doc governed vs flat
3. 100-doc governed structure
4. strict governance review benchmark

Optional but valuable:
5. QA comparison against RAG / GraphRAG
6. patched enrichment pilot
