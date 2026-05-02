# Governed Knowledge Graph: Extraction + QA-Utility Plan

> **2026-04-30 status update.** MuSiQue n=20 is locked in as the headline QA result with all three systems (MAG-KG bridge, Microsoft Official GraphRAG, Document RAG) uniformly scored on the same 20 question IDs (EM, substring-EM, token-F1, answer-rate). Numbers and table appended to `paper/collab_draft_revised.txt` as Table 4. The n=50 MuSiQue pilot file (`evaluation/results/musique_pilot_50.json`) is preserved as a strict superset of n=20 for any future scaled rerun, but the n=50 governed-KG build was cancelled because at ~15 min/doc the 99-doc rebuild would have taken ~24h and blocked the SciERC 100-doc runs. The 100-doc SciERC governed and ungoverned extractions are now running in parallel on tunnel A and tunnel B respectively (see "100-Doc SciERC Extraction" section).

We are **not** abandoning SciERC. SciERC remains the main extraction benchmark and the headline extraction table in the paper. The bigger claim of this work is that a Governed KG is a better *data structure* — a richer, domain-owned, auditable memory — and that downstream QA systems can use that memory as well as or better than document RAG and GraphRAG-style retrieval in scenarios where graph structure and domain routing matter.

## 100-Doc SciERC Extraction (running 2026-04-30 onwards)

Both runs use SciERC dev split, max-docs 100, fixed-schema, skip-evidence-linking, skip-verification, default settings (relation gleaning ON, fixed-schema pairwise scoring ON), checkpoint-every 5, resume-from-checkpoint enabled. Wall-time projection: governed ≈ 30–34 h, ungoverned ≈ 22–26 h based on the 10-doc smoke at 20.4 min/doc (governed) and the historical ungoverned pace.

| Run | Tunnel | Output JSON | Log | Governance Mode |
|---|---|---|---|---|
| Governed 100-doc | A (11434, gpu01) | `evaluation/results/scierc_governed_100.json` | `evaluation/results/n100_logs/scierc_governed_100.log` | `audit_only` |
| Ungoverned 100-doc | B (11435, gpu02) | `evaluation/results/scierc_ungoverned_100.json` | `evaluation/results/n100_logs/scierc_ungoverned_100.log` | n/a (flat) |

After both finish, score with `python scripts/analyze_scierc_build.py --kg-path <output> --split dev --max-docs 100` and compute Entity P/R/F and Triple P/R/F. The headline 100-doc extraction table will compare governed vs ungoverned on these metrics (analogous to Table 1 in the collab draft, scaled from 50 to 100 docs).

## Locked-In QA Result (MuSiQue n=20)

| System | Exact Match | Substring EM | Token F1 | Answer Rate |
|---|---|---|---|---|
| Document RAG (gemma4:31b, top-k=6) | 0.05 | 0.15 | 0.112 | 0.25 |
| Official Microsoft GraphRAG (fast/basic) | 0.35 | 0.45 | 0.410 | 1.00 |
| MAG-KG domain_basic + bridge expansion (ours) | **0.60** | **0.70** | **0.636** | 0.90 |

All three systems answer the same 20 question IDs in the same order. MAG-KG and GraphRAG answers are post-processed with the same short-answer extractor; Document RAG is prompted for short answers directly. Source files (canonical scoring): `musique_kg_qa_20_bridge_v2_final.json`, `musique_official_graphrag_fast_basic_20_final.json`, `musique_rag_20_clean_final.json`.

## What Each Benchmark Answers

| Benchmark | Question it answers | Where it appears in the paper |
|---|---|---|
| SciERC (10 / 50 / 100 docs) | Can governed KG creation extract a cleaner relational graph than flat insertion? | Headline extraction table; governance-at-scale table |
| HotpotQA (pilot, 20 ex.) | Does the governed graph capture answer-bearing facts a downstream QA system can use? | Governed-memory quality + QA-utility supporting evidence |
| **MuSiQue (pilot, 20 ex.)** | Does a governed graph help **multi-hop** QA over a structured corpus? | Primary QA-utility result |

SciERC tells us whether the *creation* pipeline is competent. Hotpot and MuSiQue tell us whether the *resulting governed graph is useful as memory* for downstream QA against document RAG, flat KG retrieval, and GraphRAG-style community retrieval. These are complementary stories — do not confuse them.

## Target Claim

The governed data structure is better when it:

1. Extracts a cleaner admitted graph than flat insertion (SciERC).
2. Captures answer-bearing information from a corpus.
3. Routes facts to meaningful domain owners with audit records and domain memory cards.
4. Supports faithful downstream QA at least as well as document RAG, flat KG retrieval, and GraphRAG-style retrieval, especially when multi-hop graph structure matters.

We do not need to claim universal QA domination. A clean result showing governance gives a measurable advantage on at least one strong axis (token F1, coverage, support precision, hallucination control, or answerability) on multi-hop QA — together with the SciERC extraction win — is the paper.

## Why SciERC F1 Alone Is Not Enough

SciERC entity/triple F1 evaluates whether extracted spans and seven fixed relation labels match an annotation scheme. That is the right metric for *extraction competence* and we keep it. But it does not measure whether the resulting KG is useful as governed memory: a QA-useful KG can contain valid answer-bearing facts that do not match SciERC's exact span boundaries or relation labels. The paper therefore keeps SciERC as the extraction benchmark and adds task-oriented graph-quality and QA-utility metrics on Hotpot and MuSiQue.

## Graph-Quality Metrics We Need

### Intrinsic KG Metrics

- Entity yield: number of non-generic entities extracted per document.
- Triple yield: number of evidence-bearing facts extracted per document.
- Orphan entity fraction: fraction of entities with no incident triples.
- Evidence-linked fact rate: fraction of triples with source/evidence metadata.
- Domain coverage: fraction of entities/triples assigned to at least one governing domain.
- Cross-domain fraction: fraction of facts spanning multiple domains.
- Governance completeness: fraction of admitted triples with governance decisions.
- Audit integrity: fraction of governance decisions with action, owner, and rationale.

### QA-Utility KG Metrics

- Gold-answer entity coverage: whether the gold answer string/alias appears as a KG entity.
- Gold-answer evidence coverage: whether the gold answer appears in triple evidence, source snippets, or domain memory.
- Question-entity coverage: whether important question entities appear in the KG.
- Supporting-document coverage: whether facts from supporting documents entered the KG.
- Domain-memory coverage: whether relevant domains have memory cards containing answer-bearing facts.

### Downstream QA Metrics

- Exact Match and token F1 for benchmarks with gold short answers.
- Faithfulness/support precision for graph-grounded answers.
- Coverage/completeness for multi-hop answers.
- Abstention correctness for unanswerable or unsupported questions.
- Error decomposition: extraction miss vs routing miss vs retrieval miss vs synthesis miss.

## Benchmark Strategy

### Primary QA-Utility Benchmark: MuSiQue

MuSiQue is the best fit for the governed-KG-as-memory story because it is explicitly designed for connected multi-hop QA, and KG structure plus governed routing should matter more there than in single-document lookup. The experiment uses a small pilot first, then scales if results look promising.

Run:

1. Prepare 20–50 MuSiQue examples and their source paragraphs.
2. Build a governed open-world KG from the supporting paragraphs.
3. Run document RAG, flat-KG retrieval (`flat_path_basic`), GraphRAG-style community retrieval (`graphrag_basic`), governed routing (`domain_basic`), and full governed QA (`domain_advanced`).
4. Evaluate with EM, token F1, answer-support coverage, faithfulness, and coverage.

### Debugging Benchmark: HotpotQA

HotpotQA is the fast-iteration loop because utilities already exist in the repo and the supporting-doc setup is small. We use it to debug extraction → routing → retrieval → synthesis end-to-end before paying MuSiQue's cost. Hotpot results may also appear in the paper as supporting evidence, but MuSiQue is the headline QA-utility result.

### Stretch Benchmark: CRAG

CRAG is newer and stronger as a RAG benchmark, but it is a larger integration project because it includes mock APIs and dynamic search. Do not block the main paper on CRAG.

## Implementation Phases

### Phase 1: Stop Losing Information During KG Creation

Status: started.

- Preserve `source_segment`, `source_segments`, and document IDs through entity coreference.
- Allow relation extraction to match entity labels and mentions, not only canonical text.
- Fall back to all entities for single-segment documents when coreference metadata is incomplete.
- Keep evidence snippets in domain memory cards.
- Ensure domain expert prompts include memory cards.
- Add a dedicated answer-bearing fact extraction pass for QA-useful values and attributes.
- Treat numbers, dates, capacities, roles, nationalities, offices, locations, aliases, and work titles as first-class answer candidates in open-world KG creation.
- Convert answer-bearing values into simple attribute facts when possible, e.g. `(arena) -[has_capacity]-> (3,677 seated)`, `(person) -[held_position]-> (Chief of Protocol)`, `(manager) -[managed_during]-> (from 1986 to 2013)`.
- Run an extraction self-check after normal extraction: ask specifically whether any answer-bearing numbers, date ranges, titles, roles, locations, or names were omitted.

Expected effect: relation extraction should no longer see zero usable entities after coreference, and QA should receive more answer-bearing evidence.

### Phase 2: Add Answer-Support Coverage Diagnostics

Status: implement next.

Create a script that reads:

- a prepared QA example file,
- a KG JSON,
- optionally an org chart,

and reports:

- gold answer in entity labels,
- gold answer in triples,
- gold answer in evidence snippets,
- question entity lexical coverage,
- domain-memory coverage,
- likely failure stage.

This tells us whether QA failure is caused by extraction, routing, retrieval, or synthesis.

### Phase 3: Add MuSiQue Prep

Status: implement next.

Create MuSiQue utilities parallel to `evaluation/hotpotqa/utils.py`:

- load raw MuSiQue JSON/JSONL,
- select answerable examples with short gold answers,
- write paragraph corpus files,
- preserve supporting paragraph metadata.

### Phase 4: Build Governed KG for MuSiQue

Use open-world, QA-permissive settings:

- no fixed SciERC schema,
- permissive governance or triage with high rejection threshold only for malformed facts,
- evidence linking enabled when runtime permits,
- memory cards refreshed after every document.

### Phase 5: Run QA Systems

Compare:

- `rag_basic`: document RAG over source paragraphs,
- `flat_path_basic`: one global KG expert over all triples,
- `graphrag_basic`: community-summary retrieval over the KG,
- `domain_basic`: governed routing to domain experts,
- `domain_advanced`: routing plus exploration/critic/provenance.

### Phase 6: Error Breakdown

For every failed question, classify:

- Extraction miss: answer never entered KG or domain memory.
- Routing miss: answer exists but in unrouted domain.
- Retrieval miss: routed domain had answer but QA context omitted it.
- Synthesis miss: context contained answer but final response was wrong.

This is the most important debugging loop for improving the actual data structure.

## Immediate Commands After Implementation

Fast Hotpot diagnostic:

```bash
PYTHONPATH=. ./.venv/bin/python -u scripts/evaluate_qa_support_coverage.py \
  --examples-json evaluation/results/hotpotqa_pilot_20.json \
  --kg-path evaluation/results/hotpotqa_governed_kg_20_supporting_openworld.json \
  --org-chart evaluation/results/hotpotqa_governed_kg_20_supporting_openworld_org.json \
  --output evaluation/results/hotpotqa_support_coverage_openworld.json
```

MuSiQue prep, after raw dataset is available:

```bash
PYTHONPATH=. ./.venv/bin/python -u scripts/prepare_musique_pilot.py \
  --source-json evaluation/results/musique_ans_dev.json \
  --n 20 \
  --output evaluation/results/musique_pilot_20.json \
  --docs-dir evaluation/results/musique_pilot_20_docs \
  --manifest evaluation/results/musique_pilot_20_docs_manifest.json
```

## Paper Framing

The final results section is structured around three claims, in order:

1. **SciERC extraction.** Governed KG creation extracts a cleaner relational graph than flat insertion (higher triple F1, lower hallucination, comparable entity F1) at 10 and 50 documents, with structural change (low Jaccard) and acceptable wall-clock overhead.
2. **Governed memory quality.** The resulting KG is not just a set of triples: it carries domain ownership, audit records, evidence snippets, and domain memory cards, with high governance completeness, audit integrity, and domain coverage at 100 documents.
3. **QA utility.** On multi-hop QA (MuSiQue, with HotpotQA as supporting), the governed graph is at least as useful as document RAG, flat-KG retrieval, and GraphRAG-style retrieval, with a measurable advantage on at least one of {token F1, coverage, support precision, faithfulness, abstention correctness}.

We do not claim SOTA SciERC extraction, and we do not claim universal QA domination. The contribution is the governed *data structure* and the evidence that it pays off on both extraction quality and downstream QA utility.
