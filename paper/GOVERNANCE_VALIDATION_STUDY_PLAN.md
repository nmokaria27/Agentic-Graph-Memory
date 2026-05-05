# Governance Validation Study Plan

This plan turns the current governed-vs-flat result into an explicit validation of governance. The main paper claim is not "our extractor is SOTA." The claim is that governed admission changes what enters the graph, creates auditable ownership, and improves downstream use. These studies isolate that claim.

## Current Status

### Extraction Table

The current paper table is based on Gemma/Ollama SciERC fixed-schema runs:

- 10-doc flat vs governed is complete.
- 50-doc flat vs governed is complete.
- Human-review samples for the 50-doc Gemma table have been generated.
- GPT-5 flat 50-doc is complete.
- GPT-5 governed 50-doc is still running/resuming from checkpoint and should be scored when complete.

The active GPT-5 governed run is the final extraction-table dependency. It uses triage governance with a fixed-schema confidence floor:

```bash
LLM_BACKEND=openai \
OPENAI_API_KEY="$OPENAI_API_KEY_BACKUP" \
PYTHONPATH=. \
python -u scripts/build_governed_scierc.py \
  --split test \
  --max-docs 50 \
  --model gpt-5 \
  --fixed-schema \
  --fixed-schema-min-admission-confidence 0.75 \
  --skip-evidence-linking \
  --skip-verification \
  --governance-mode triage \
  --output evaluation/results/scierc_governed_gpt5_test_50_schemahard.json \
  --org-output evaluation/results/scierc_governed_gpt5_test_50_schemahard_org.json \
  --stats-output evaluation/results/scierc_governed_gpt5_test_50_schemahard_stats.json \
  --checkpoint-every 1 \
  --resume-from-checkpoint
```

Important limitation: this run skips EvidenceLinker and VerificationAgent, so it can support the governed-vs-flat construction table, but it cannot by itself support the professor's EvidenceLinker-only ablation.

### Human Review Samples

Generated refreshed 50-doc Gemma triage samples:

- `evaluation/results/governance_ablation_samples_50docs_triage_refresh.json`
- `evaluation/results/governance_ablation_samples_50docs_triage_refresh.csv`

Current sample populations:

- Governed triples: 221
- Flat triples: 407
- Governed-only: 112, sampled 100
- Flat-only: 298, sampled 100
- Shared: 109, sampled 50
- Revised: 22, sampled all 22

If GPT-5 becomes the main extraction table, regenerate the same samples from the final GPT-5 governed and flat artifacts.

## Validation Study 1: Governance Ablation

The goal is to show whether governance adds value beyond simpler alternatives. Each ablation should use the same document set, model, fixed SciERC schema, entity extraction, and relation extraction wherever possible. Only the admission mechanism changes.

### Main Baseline: Full MaKG Governance

This is the existing governed KG condition.

Pipeline:

1. Extract entities and triples.
2. Assign entities to domains.
3. Route triples by ownership.
4. Apply triage governance.
5. Use domain reviewer for risky triples.
6. Commit approved/revised triples and audit all decisions.

Metrics:

- Entity F1
- Triple strict F1
- Triple mapped/fuzzy F1
- Triple hallucination rate
- Triple count
- Zero-triple docs
- Decision counts
- Revised count
- Cross-domain fraction

Interpretation:

This is the only condition with full governance: domain ownership, routing, review, revision, and audit.

### Q1: Evidence/Verification-Only Flat KG

Question:

Is MaKG better only because EvidenceLinker and VerificationAgent filter unsupported triples?

Pipeline:

1. Extract entities and candidate triples.
2. Run EvidenceLinker.
3. Run VerificationAgent.
4. Discard triples without source support or verification.
5. Insert remaining triples into a flat KG.
6. Do not assign domains.
7. Do not route triples.
8. Do not run governance review.

Metrics:

- Triple F1
- Triple mapped/fuzzy F1
- Triple hallucination rate
- Triple count
- Zero-triple docs
- Source-supported triple count
- Source-supported triple fraction

Expected interpretation:

If Evidence/Verification-only is close to MaKG, governance is mostly a support filter. If MaKG beats it, governance is doing more than evidence filtering.

Implementation note:

This cannot be honestly computed from current GPT-5 schemahard artifacts because those runs used `--skip-evidence-linking --skip-verification`. We need a dedicated evidence-enabled run, preferably first on 10 docs, then 50 docs only if runtime/credits are acceptable.

Recommended run:

```bash
LLM_BACKEND=openai \
OPENAI_API_KEY="$OPENAI_API_KEY_BACKUP" \
PYTHONPATH=. \
python -u scripts/build_ungoverned_scierc.py \
  --split test \
  --max-docs 10 \
  --model gpt-5 \
  --fixed-schema \
  --output evaluation/results/scierc_evidence_only_flat_gpt5_test_10.json \
  --stats-output evaluation/results/scierc_evidence_only_flat_gpt5_test_10_stats.json \
  --checkpoint-every 1
```

Then rerun at 50 docs only if the 10-doc result is informative and affordable.

### Q2: Global LLM Reviewer, No Domains

Question:

Could governance be replaced by a single non-domain LLM judge?

Pipeline:

1. Use the same extracted proposal stream as full MaKG.
2. Ignore domain ownership and org chart.
3. Send each candidate triple plus evidence to a global LLM reviewer.
4. Reviewer decides approve/reject/revise.
5. Insert approved/revised triples into a flat KG.

Metrics:

- Triple F1
- Triple mapped/fuzzy F1
- Triple hallucination rate
- Triple count
- Zero-triple docs
- Approve/reject/revise counts
- Revision precision from human sample if available

Expected interpretation:

If global reviewer matches MaKG, then domain ownership is not necessary. If MaKG beats it or produces better human-validated revisions, then governance is not just "LLM judge as filter."

Implementation note:

This can be implemented cheaply by replaying the audit/proposal stream from the governed artifact instead of re-running extraction. That isolates admission policy from extraction.

Recommended division of work:

- Codex: implement replay script and scoring.
- Neel: review/adjust the global-reviewer prompt for fairness.

### Q3: Domain Ownership Without LLM Governance

Question:

Does domain ownership help even without an LLM reviewer?

Pipeline:

1. Use the same extracted proposal stream as full MaKG.
2. Keep entity-domain assignments.
3. Classify each triple as single-owner, cross-domain, or unowned.
4. Apply deterministic rules:
   - single-owner: approve
   - cross-domain: approve if both endpoints have stable owners
   - unowned: reject or escalate-but-do-not-admit
5. No LLM reviewer.
6. Insert approved triples into a flat KG.

Metrics:

- Triple F1
- Triple mapped/fuzzy F1
- Triple hallucination rate
- Triple count
- Zero-triple docs
- Counts by assignment type

Expected interpretation:

This tests the structural value of ownership separately from reviewer intelligence. If this beats flat, ownership/routing itself is useful. If full MaKG beats this, the LLM review/revision layer adds value on top of ownership.

Implementation note:

This is the cheapest ablation. It can be replayed from the governed audit log with no new LLM calls.

### Proposed Study 1 Table

Use one table for the paper:

| Condition | Domains? | LLM Review? | Evidence Filter? | Triple F1 ↑ | Mapped/Fuzzy F1 ↑ | Hallucination ↓ | Triples | Zero Docs ↓ | Supported Triples ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat KG | No | No | No | TBD | TBD | TBD | TBD | TBD | TBD |
| Evidence-only flat | No | No | Yes | TBD | TBD | TBD | TBD | TBD | TBD |
| Global LLM reviewer | No | Yes | Yes/available | TBD | TBD | TBD | TBD | TBD | TBD |
| Domain rules only | Yes | No | Same proposals | TBD | TBD | TBD | TBD | TBD | TBD |
| Full MaKG governance | Yes | Yes | Same proposals | TBD | TBD | TBD | TBD | TBD | TBD |

The key contrast is not a single score. It is the pattern:

- Evidence-only tests whether support filtering explains the result.
- Global reviewer tests whether a generic LLM judge explains the result.
- Domain-rules-only tests whether ownership has independent value.
- Full MaKG should be best or most balanced on F1/hallucination/revision quality.

## Validation Study 2: Human Review of Triples

The goal is external validation of the graph differences. This directly answers reviewer skepticism about low strict F1 and whether "unique-to-governed" facts are real.

### Sampling Groups

Use four groups:

- A: Governed-only triples, up to 100.
- B: Flat-only triples, up to 100.
- C: Shared triples, 30-50.
- D: Revised triples, all revised triples.

### Annotation Fields

Each sampled row already contains blank columns for:

- `annotation_supported_by_source`: yes/no/partial
- `annotation_correct_relation`: yes/no/partial
- `annotation_correct_endpoints`: yes/no/partial
- `annotation_useful_for_graph_or_qa`: yes/no
- `annotation_decision_quality`: good/bad/unclear
- `annotation_notes`: free text

### Human Instructions

For each triple, inspect the subject, relation, object, and evidence/source text.

Mark `supported_by_source=yes` only if the source explicitly supports the relation between the endpoints.

Mark `correct_relation=yes` only if the relation label is semantically correct under the SciERC schema.

Mark `correct_endpoints=yes` if the subject/object are the right scientific concepts, even if surface canonicalization differs slightly.

For flat-only triples, `decision_quality=good` means governance was right to exclude it. For governed-only triples, `decision_quality=good` means governance admitted a useful supported triple that flat missed. For revised triples, `decision_quality=good` means the revision improves the original proposal.

### Proposed Study 2 Table

| Group | Sample Size | Source Supported ↑ | Correct Relation ↑ | Correct Endpoints ↑ | Useful for Graph/QA ↑ | Good Governance Decision ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Governed-only | 100 | human | human | human | human | human |
| Flat-only | 100 | human | human | human | human | human |
| Shared | 50 | human | human | human | human | control |
| Revised | all | human | human | human | human | human |

Expected paper use:

This table is more important than squeezing strict SciERC F1. It shows whether governance decisions are qualitatively correct and whether revised triples are actually better.

## Validation Study 3: Cost Analysis

The cost analysis should separate extraction cost from governance cost.

Report:

- Wall-clock time per document.
- LLM calls per document.
- Prompt tokens per document.
- Completion tokens per document.
- Estimated API dollars per document.
- Incremental governance overhead over flat.
- Cost per admitted triple.
- Cost per supported/correct triple if human review is available.

Proposed table:

| System | Model | Docs | Wall Time | LLM Calls | Tokens | Cost | Triples | Cost/Triple |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | GPT-5 | 50 | TBD | TBD | TBD | TBD | TBD | TBD |
| MaKG | GPT-5 | 50 | TBD | TBD | TBD | TBD | TBD | TBD |
| Flat | Gemma/Ollama | 50 | TBD | TBD | local | $0 API | TBD | $0 API |
| MaKG | Gemma/Ollama | 50 | TBD | TBD | local | $0 API | TBD | $0 API |

Use the existing usage logs for GPT-5:

- `evaluation/results/gpt5_scierc_ungov_50_usage.jsonl`
- `evaluation/results/gpt5_scierc_gov_50_schemahard_policy075_usage.jsonl`
- `evaluation/results/gpt5_scierc_gov_50_schemahard_policy075_resume_usage.jsonl`

## Validation Study 4: Domain Validation

This is optional unless the professor explicitly asks for it, but the current data can support a lightweight version.

Questions:

- Are most entities assigned to a domain?
- Are domains stable and interpretable?
- Are cross-domain triples meaningful?
- Does routing send triples to plausible owners?

Metrics:

- Number of domains.
- Entity domain coverage.
- Cross-domain fraction.
- Assignment type counts.
- Domain decision counts.
- Human-validity sample of entity-domain assignments.

For SciERC fixed-schema runs, the current domain structure is intentionally three broad domains:

- `methods_and_systems`
- `tasks_and_concepts`
- `resources_and_evaluation`

This is not bad. For benchmark mode, broad schema-derived domains are more defensible than LLM-created microdomains because they make the ownership function stable and reproducible.

## GPT-5 Extraction Diagnosis

Current GPT-5 behavior is not "worse intelligence." It is a precision problem.

Observed pattern:

- GPT-5 finds more candidate facts and often has better partial entity coverage.
- It over-extracts many plausible-but-not-SciERC triples.
- Strict SciERC relation scoring is brittle to canonicalization (`used_for` vs `Used-for`) unless relation labels are normalized.
- Governance improves this by canonicalizing and rejecting low-confidence proposals, but the current reviewer is still permissive.

What to do:

1. Let the current GPT-5 governed 50-doc run finish.
2. Score strict and mapped/fuzzy metrics.
3. Run post-hoc threshold replay at 0.75, 0.80, 0.85 for diagnosis only.
4. Use 0.75 as the main governed policy unless 0.80 clearly improves F1 without destroying recall.
5. Do not rerun GPT-5 flat unless the final table requires a flat run under exactly the same confidence policy. The completed flat artifact is enough for now if scoring canonicalizes relations.

Do not silently postprocess only the governed side. If relation canonicalization is used, apply it to both flat and governed in the evaluator and state that all SciERC relations are mapped to the fixed schema before scoring.

## Immediate Execution Order

1. Keep the active GPT-5 governed 50-doc run alive until it completes.
2. Score GPT-5 governed 50-doc against SciERC after completion.
3. Regenerate human-review samples from GPT-5 governed vs GPT-5 flat if GPT-5 becomes the main table.
4. Implement replay ablations for Q2 and Q3 from the audit log.
5. Run Q2/Q3 first on Gemma 50-doc because all artifacts are complete.
6. Run Q2/Q3 on GPT-5 after final governed artifact exists.
7. Run Q1 evidence-only on 10 GPT-5 docs as a smoke test.
8. Only run Q1 at 50 docs if the 10-doc result changes the story enough to justify cost.

## Delegation

Codex:

- Monitor and score GPT-5 governed extraction.
- Generate human-review CSV/JSON.
- Implement and run deterministic domain-rules ablation.
- Implement scoring and cost table extraction.

Neel:

- Review the global LLM reviewer prompt for Q2.
- Help decide whether Q2 should be GPT-5-only or Gemma + GPT-5.

User/professor:

- Complete human annotations for Study 2.
- Decide whether Study 4 domain validation should be a main paper table or appendix.

## Recommended Paper Framing

Keep the main extraction table as governed vs flat. Add the ablation table directly after it.

The text should say:

"The governed graph is not merely a flat graph with unsupported triples removed. Evidence-only filtering, global LLM review, and deterministic ownership rules isolate different parts of the system. Full governance is the only condition that combines ownership-aware routing, review, revision, and auditable admission."

Use human review to support the claim:

"Because strict SciERC matching underestimates semantically valid canonicalization variants, we additionally perform human review over governed-only, flat-only, shared, and revised triples."

