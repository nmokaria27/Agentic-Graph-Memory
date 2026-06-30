# Extraction Critic & Self-Improvement Loop — Prompts

Drop-in prompts for a reflection layer that (1) catches degenerate triples the
verifier currently waves through, and (2) distills *reusable lessons* that get
injected back into the EntityExtractor / RelationExtractor on the next run.

Loop shape:

```
extract → CRITIC (audit + correct + distill lessons) → write lessons to memory
        → next extraction injects LESSONS_BLOCK into extractor/relation prompts
```

The CRITIC is a single agent run per document (or per batch of N triples). It is
cheap to add because it reuses your existing triple schema
(`subject / subject_id / relation / object / object_id / confidence / evidence`)
and your blackboard/shared-memory plumbing.

---

## 1. CRITIC system prompt

```
You are the Extraction Critic for a governed knowledge-graph pipeline. You do
NOT extract new knowledge. Your only job is to find DEGENERATE and LOW-VALUE
triples that earlier agents produced, correct or drop them, and write down
GENERALIZABLE LESSONS so the extractor stops repeating the same mistakes.

You are graded on PRECISION OF YOUR JUDGEMENTS. Flag a triple only when it
clearly matches one of the failure modes below. When a triple is fine, say so
and move on — do not invent problems.
```

## 2. CRITIC task prompt (per document / batch)

```
SOURCE TEXT:
{text}

TRIPLES UNDER REVIEW (each: subject | relation | object | confidence | evidence):
{triples_json}

PRIOR LESSONS (mistakes already learned — do not re-derive these, just apply them):
{prior_lessons}

Audit every triple against this FAILURE TAXONOMY. Assign at most one primary
failure code per triple; "OK" if none apply.

  SELF_REF      subject and object are the same entity (same id, same surface,
                or trivial case/whitespace variant). e.g. (IL-6, assoc_with, IL-6).
  CONTAINMENT   object is a substring/superset of subject or vice-versa, so the
                triple states nothing new. e.g. (diabetes, is_a, type 2 diabetes),
                (apple_inc, created_product, apple_i) where one merely re-names the other.
  TAUTOLOGY     the relation just restates the entity name or is true by definition;
                carries no fact. e.g. (heart_disease, is_disease, disease).
  CIRCULAR      a pair of triples that re-encode each other with inverted/duplicate
                relations and add no information. e.g. (A, part_of, B) AND
                (B, has_part, A) when only one is meaningful.
  GENERIC_NODE  subject or object is a generic placeholder, not a real entity:
                "the study", "the results", "the method", "these patients", bare
                pronouns, bare numbers without unit/meaning, lone adjectives.
  PARAPHRASE_REL the relation label is a vague paraphrase of the sentence rather
                than a typed relation: "related to", "associated with the topic of",
                "is connected to", "discussed alongside".
  UNGROUNDED    the evidence span does not actually license this subject-relation-
                object; the relation is co-occurrence or plausibility, not stated.
  DUP           a near-duplicate of another triple already in this batch (same
                normalized s/r/o) — keep the highest-confidence one, mark the rest DUP.

For each triple return a verdict. When the underlying fact is real but the triple
is malformed (wrong direction, paraphrased relation, object too broad), prefer
CORRECT over DROP and supply the fixed triple. When there is no salvageable fact,
DROP it.

Then DISTILL LESSONS. A lesson is a short, GENERAL rule — not a restatement of one
triple — that would have prevented the mistake and will generalize to unseen text.
Write lessons as imperative guidance the extractor can follow, each with one
concrete BAD example and its GOOD counterpart drawn from this batch. Merge with
PRIOR LESSONS rather than duplicating them; only emit lessons that are new or
sharper than what already exists. Cap at 5 lessons per run.

Return JSON only:
{
  "verdicts": [
    {
      "subject": "...", "relation": "...", "object": "...",
      "decision": "OK | CORRECT | DROP",
      "failure_code": "OK | SELF_REF | CONTAINMENT | TAUTOLOGY | CIRCULAR | GENERIC_NODE | PARAPHRASE_REL | UNGROUNDED | DUP",
      "reason": "<one line, cite the evidence span>",
      "corrected_triple": null | {
        "subject": "...", "subject_id": "...", "relation": "...",
        "object": "...", "object_id": "...", "confidence": <0.0-1.0>,
        "evidence": "<exact span from SOURCE TEXT>"
      }
    }
  ],
  "lessons": [
    {
      "id": "<short_snake_case_key, stable across runs>",
      "rule": "<imperative, general — e.g. 'Never emit a triple whose object only re-names the subject with added qualifiers'>",
      "failure_code": "<code this lesson guards against>",
      "bad_example": "(subject, relation, object) that violated the rule",
      "good_example": "(subject, relation, object) or DROP — the correct handling"
    }
  ],
  "batch_stats": {
    "reviewed": <int>, "ok": <int>, "corrected": <int>, "dropped": <int>,
    "degeneracy_rate": <dropped+corrected / reviewed, 0.0-1.0>
  }
}
```

## 3. LESSONS_BLOCK — injected into the extractor next run

Render the top lessons (deduped by `id`, most-violated first) into this block and
append it to `COMBINED_EXTRACTION_PROMPT` and `TAIL_BINDING_PROMPT`. This is the
mechanism that closes the loop — the extractor literally reads its own past
mistakes before producing new output.

```
LESSONS FROM PAST MISTAKES — these are real errors this pipeline made before.
Do not repeat them. Each rule has a BAD case that was rejected and the GOOD fix.
{for each lesson}
- {rule}
    BAD:  {bad_example}
    GOOD: {good_example}
{end}
Before emitting any triple, check it against every rule above. If a triple
matches a BAD pattern, drop it or rewrite it to the GOOD form.
```

---

## 4. Integration notes (how this becomes "self-improving")

1. **Run order.** Insert the Critic *after* RelationExtractor, before/replacing the
   recall-biased Verifier — or run it as a second Verifier pass. It reads triples,
   writes back `corrected_triple`s, and drops the rest.

2. **Persist lessons.** Store the `lessons` array in `SharedMemory` (semantic tier)
   or a `lessons.json` keyed by `id`. On each new document, load the top-K lessons
   (rank by how often each `failure_code` fired recently) and fill `{prior_lessons}`
   and `{LESSONS_BLOCK}`. This is the only state the loop needs.

3. **Close the recall-bias hole in the Verifier.** The current
   `VERIFICATION_PROMPT` says "accept if consistent with the text, even if not
   explicitly stated." Add: "A triple that is trivially true (object restates
   subject, relation restates entity, pure co-occurrence) is NOT valuable — reject
   it even if consistent." Otherwise the Critic and Verifier will fight.

4. **Add the code-level guard the prompts can't fully cover.** Extend the existing
   `invalid_self_refs` filter in `relation_extractor.run()` to also drop CONTAINMENT
   (normalized `subj in obj or obj in subj`) and exact-normalized DUP. Cheap,
   deterministic, no LLM call — catches the long tail.

5. **Measure it (reward signal).** Add a gold-free `degeneracy_rate` metric to
   `evaluation/evaluate_kg.py` (fraction of triples matching SELF_REF / CONTAINMENT /
   PARAPHRASE_REL / GENERIC_NODE by rule). Track it across runs — a working loop
   should show it trending toward zero as the lesson bank grows. Reuse KGAFE's
   entity normalization so the metric and the Critic agree on "same entity."

6. **Guardrails so the loop doesn't over-correct.** Cap lessons (e.g. 30 total,
   evict least-fired). Require a lesson to fire ≥2 times before it's injected, so
   one-off noise doesn't permanently bias the extractor toward under-extraction.
```
