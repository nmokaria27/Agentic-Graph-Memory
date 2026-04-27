# Paper and Pipeline Fact Check

This note cross-checks the current paper draft against the repository.

## Major findings

- The extraction pipeline in code has eight named agents, but the default orchestrator currently skips the explicit `ExtractionValidator` stage and consolidates quality control into `ExtractionVerificationAgent`.
- The paper draft says document segmentation targets about 512 tokens. The implemented `DocumentProcessor` actually segments by character length, using roughly 1500 to 2000 characters with overlap.
- The current repo includes a substantial incremental enrichment path (`IncrementalEnricher` + `ConflictResolver`) that is not written out in the paper draft.
- The paper draft reports 33 relation types in the biomedical graph. The saved `kg_export.json` artifact contains 35 unique relation types.
- The paper draft’s QA results table does not match the saved fixed KGAFE artifact. The repository’s `kgafe_results_fixed.json` contains different per-question rows and should be the source of truth if those are the numbers you want to publish.
- The paper draft says most verified facts came from Tier 3 semantic verification. In the fixed KGAFE artifact, most supported facts come from Tier 2 path-based verification.

## Extraction pipeline checks

- `DocumentProcessor` exists and stores segmented documents plus metadata in shared memory.
- `DomainClassifier` supports both dynamic schema generation and fixed-schema injection.
- `EntityExtractor` supports multi-stage extraction, coreference-style consolidation, alias registration, and optional self-consistency.
- `RelationExtractor` supports RHF extraction, discovered relation types, and a connectivity pass.
- `EvidenceLinker` computes final confidence using the weighted formula described in the draft.
- `DeliberationCoordinator` implements seven-level voting, consensus thresholding, and debate triggering.
- `ExtractionVerificationAgent` is the effective last quality gate in the shipped orchestrator.
- `KnowledgeOrganizer` handles aliasing, cleanup, and graph integration.

## QA checks

- `DomainBuilder` and `OrgChart` are implemented.
- `AdvancedQAOrchestrator` includes the five major QA features described in the paper: active exploration, debate, critic, session memory, and provenance.
- The QA server caches the org chart, which supports the claim that the structure can be serialized and reused.
- The critic prompt checks for hedging, but hedging is not exposed as a separate structured issue type in the JSON schema.

## KGAFE checks

- `AtomicDecomposer`, `TripleVerifier`, `JudgePanel`, and `BenchmarkGenerator` are all implemented.
- Judge panel weights and overall pass logic match the draft.
- The current evaluator computes `kg_faithfulness` and `kg_precision` the same way: supported facts divided by total facts.
- `kgafe_results_fixed.json` supports the headline claim of 96/97 supported facts and average KGAFE 0.8981.
- `demo_results.json` does not support that claim. Its aggregate KGAFE score is much lower, so the paper should explicitly reference the fixed artifact if that is the published result.

## Bibliography cleanup

- The cited KARMA entry in the LaTeX draft does not match the paper title I was able to verify. The verified paper title is `KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment`.
- The draft cites `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena` as NeurIPS 2024, but the verified paper is associated with NeurIPS 2023.
- The draft cites `Active Retrieval Augmented Generation` as EMNLP 2024, but the verified paper is associated with EMNLP 2023.
- I could not cleanly verify the exact `MiroFish` reference used in the draft. That citation should be replaced with a confirmed paper before submission.

## Artifact-backed numbers to use

From `kg_export.json`:

- 183 entities
- 159 triples
- 10 entity types
- 35 relation types

From `evaluation/results/metrics_3docs_fixed_v3.json`:

- Entity F1 strict: 0.5577
- Entity F1 partial: 0.6667
- Entity type accuracy: 0.7586
- Mapped entity type accuracy: 0.7931
- Triple F1 strict: 0.0408
- Triple F1 mapped/fuzzy: 0.0816

From `kgafe_results_fixed.json`:

- 4 evaluated questions
- Average KGAFE: 0.8981
- 97 total atomic facts
- 96 supported
- 0 contradicted
- 1 unverifiable

## Recommendation

If the paper is meant to describe the repository as it currently exists, the minimum revision set is:

- Update the extraction-method text so it matches the actual orchestrator.
- Add a short subsection on incremental enrichment.
- Refresh both result tables from the saved artifacts.
- Clarify which artifact is the source of the headline KGAFE number.
