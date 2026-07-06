# RESEARCH_IDEAS.md — literature-grounded improvement ideas (2026-07-06)

## How to read this doc

Every idea passed the project's constraint filter: local-only (Nemotron-30B FP8 on vLLM + mxbai-embed-large on Ollama), domain-adaptive (no fixed extractors, schema discovered per corpus), no benchmark-vocabulary tuning, implementable in days on this codebase. Each idea names the pipeline stage or QA component it touches, the measured failure mode it targets, and the phase it serves (A-residue / B = LongMemEval+MAB / C = MuSiQue/2Wiki / D = self-improvement), plus a pre-registerable experiment. Ideas that overlap something already in the repo are written as a *delta from current state*, not greenfield; techniques that fail the filter get one line in §6.

---

## 1. Extraction ideas

### 1. Guided JSON output via vLLM structured decoding (thinking channel left free)

- **Source:** vLLM structured outputs (xgrammar/Outlines backends) — [docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html), [vLLM blog on structured decoding](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html); caveat evidence: "Let Me Speak Freely?" ([arXiv:2408.02442](https://arxiv.org/abs/2408.02442), 2024).
- **What it is:** Pass `guided_json`/`response_format` with a JSON schema on extraction calls so the *answer* channel is grammar-constrained token-by-token. Critically, on reasoning models the constraint applies only after the think block closes — Nemotron's hidden reasoning stays unconstrained, so the known failure mode from "Let Me Speak Freely" (format constraints degrade reasoning) is largely sidestepped: the model reasons freely, then emits schema-valid JSON that cannot be malformed.
- **Evidence:** SqueezeBits benchmarking: unguided schema-conformance 90–94% vs >96–98% guided; xgrammar adds low per-token overhead with grammar caching. "Let Me Speak Freely" reports measurable reasoning degradation under strict JSON — mixed evidence, hence the free-thinking-channel design. No published numbers specifically for Nemotron.
- **Fit:** Stages 2–4 + gleaning + stage 9 (`multi_agent_kg/llm/openai_client.py`). Directly attacks the residue of the "122→9" class: JSON parse failures, truncated/malformed output after ladder rescues, and wasted ladder climbs (a truncated guided response is still parseable prefix-free garbage → detectable, and finish_reason semantics stay clean). Also a prerequisite that makes SC re-validation (idea 6) cheaper — no parse-failure samples.
- **Experiment:** After Phase 4: add `guided_json` behind an env flag (`EXTRACTION_GUIDED_JSON=1`), re-run hybrid on slice A + held-out slice B, singlepass control untouched. Success bar: JSON-parse-failure count → 0, ladder invocations not increased, entR/pairR/relF1 within noise of v5 (this is a robustness fix, not a recall fix — pre-register it as such).
- **Effort:** S–M (client change + schema definitions per stage; verify the vLLM build on gpu02 supports structured output with the reasoning parser).

### 2. Targeted low-yield re-glean ("MANY entities were missed" loop)

- **Source:** Microsoft GraphRAG extraction gleaning (CONTINUE/LOOP prompts) — [GraphRAG paper, arXiv:2404.16130](https://arxiv.org/abs/2404.16130), [repo prompts](https://github.com/microsoft/graphrag); already flagged as "candidate future fix" in ROADMAP §7.
- **What it is:** After the wide-harvest call, compare yield against a domain-general expectation (e.g., entities per 100 tokens of segment, computed from the corpus's own running stats — no benchmark vocabulary). If yield is anomalously low, issue one bounded re-glean call: "MANY entities and relationships were missed in the last extraction; emit ONLY additional ones" with the previous output in context. GraphRAG runs gleaning unconditionally; the delta here is *trigger only on low yield*, so cost concentrates exactly on the reasoning-runaway docs where the wide call under-yields.
- **Evidence:** GraphRAG ships gleaning as a core mechanism and documents that "more information is extracted when performing multiple passes," but publishes no clean ablation — weak published evidence. Strong *internal* evidence: doc 33 hybrid 7/15 vs singlepass 14/15 entities, doc 1 repeated ladder rescues (MATRIX_REPORT v4 root-cause; ROADMAP §7).
- **Fit:** Stage 3 wide-harvest front end. Targets the last known systematic hybrid-vs-singlepass gap: reasoning-runaway docs where wide-call yield collapses (the residual slice-B entR deficit).
- **Experiment:** Post-Phase 4: implement yield trigger + single re-glean, re-run hybrid on slice B (contains doc 33) + slice A, singlepass control untouched. Success bar (pre-registered): doc-33-class docs recover ≥50% of their singlepass entity deficit; aggregate slice-B hybrid entR +0.05 or better; ≤1 extra LLM call per triggered doc; no pairR/relF1 regression; trigger fires on ≤30% of docs.
- **Effort:** S–M.

### 3. Atomic-fact decomposition front end (ATOM-style)

- **Source:** ATOM: AdapTive and OptiMized dynamic temporal KG construction ([arXiv:2510.22590](https://arxiv.org/abs/2510.22590), EACL 2026 Findings); related: iText2KG's Document Distiller ([arXiv:2409.03284](https://arxiv.org/abs/2409.03284), WISE 2024).
- **What it is:** Insert a cheap LLM pass that rewrites each segment into minimal, self-contained atomic facts (pronouns resolved, one predication per line), then run entity/relation extraction over the atomic-fact list instead of raw prose. Extraction prompts become short and uniform, which also suppresses reasoning ramble; ATOM additionally extracts per-fact temporal 5-tuples (see idea 19/20).
- **Evidence:** ATOM reports ~18% higher extraction exhaustivity and ~33% better run-to-run stability vs baseline LLM KG construction, plus large latency wins from parallel merging. Stability is the headline for us — rhf/hybrid variance (±0.08 entR at n=5) is a documented internal problem.
- **Fit:** Stage 1/3 boundary (DocumentProcessor emits atomic facts; EntityExtractor consumes them). Targets: run-to-run variance (LESSONS Lesson 5), reasoning runaway on pathological prose (short inputs → less runaway), and implicitly coref (atomic rewriting resolves pronouns *before* stage-4 coref sees them, shrinking its job).
- **Experiment:** Post-Phase 4, behind a flag: hybrid+atomic vs frozen hybrid v2 on slices A+B, 2 repeats each to measure variance. Success bar: entR within noise or better AND cross-repeat entR spread halved; wall-time increase ≤1 extra call per segment.
- **Effort:** M (1–2 days: one new prompt + plumbing + re-run).

### 4. Definition-based canonicalization for coref (EDC-style "define" step)

- **Source:** Extract–Define–Canonicalize ([arXiv:2404.03868](https://arxiv.org/abs/2404.03868), EMNLP 2024).
- **What it is:** Before merging two extracted entities, have the extractor emit (or a batched call generate) a one-sentence *definition* of each entity from its evidence; embed the definitions with mxbai and require definition-similarity above a threshold — surface-form similarity alone is not enough. EDC does this for schema components; applied here to entity coref, it converts "these strings look alike" into "these descriptions denote the same thing."
- **Evidence:** EDC shows definition-embedding canonicalization outperforms prior canonicalization on WebNLG/REBEL/Wiki-NRE in both schema-present and schema-free settings (peer-reviewed, with ablations of the define step). No numbers for the exact coref use, so treat as principled transfer.
- **Fit:** Stage 3 sub-stage 4 (coreference) — the documented coref over-merge that costs hybrid ~0.09–0.20 entR vs its own wide harvest (MATRIX_REPORT v3 residual-gap analysis) and once collapsed 29→1 (fixed structurally, but eager merging remains). Fully domain-adaptive: definitions come from the corpus itself.
- **Experiment:** Post-Phase 4: gate merges on definition-sim ≥ τ (sweep τ on slice A only), evaluate on slice B. Success bar (the standing Phase-A bar): hybrid entR → singlepass's level (~0.81 A / 0.88 B) without pairR/relF1 regression; count of merges rejected by the gate logged per doc.
- **Effort:** M.

### 5. LLM merge-verification gate on coref clusters (KGGen/CORE-KG-style)

- **Source:** KGGen's LLM-validated iterative clustering ([arXiv:2502.09956](https://arxiv.org/abs/2502.09956), NeurIPS 2025); CORE-KG's type-scoped coref with alias caches ([arXiv:2506.21607](https://arxiv.org/abs/2506.21607), 2025).
- **What it is:** Two cheap guards on stage-4 coref output: (a) every proposed merge group above size 2 gets one batched yes/no LLM check ("do ALL of these mentions denote the same real-world entity? If not, split") — KGGen validates every cluster this way; (b) merges are scoped by the *system's own* discovered entity types (type disagreement blocks a merge unless the LLM check overrides), with per-type alias→canonical caches as in CORE-KG. Complementary to idea 4 (embedding evidence) — this is the LLM veto.
- **Evidence:** KGGen +18% over OpenIE/GraphRAG on its MINE benchmark (self-published benchmark — moderate evidence). CORE-KG: node redundancy 30.4%→20.3% (33% relative) with type-scoped canonicalization — directionally exactly our over-merge/under-merge tradeoff.
- **Fit:** Stage 3 coref + stage 9 dedupe. Same failure as idea 4 (coref over-merge). Delta from current state: coref already can't delete; this makes it *justify* merges. Also fixes the SC-union near-duplicate-variant hazard (EXTRACTION_EXPERIMENTS: unioned variants confuse coref).
- **Experiment:** Same protocol as idea 4; run ideas 4 and 5 as separate arms then combined, one loop each on slice A, verdict on slice B. Success bar identical. Instrument: merges proposed / vetoed / split.
- **Effort:** S (guard b) + M (guard a).

### 6. Self-consistency re-validation done right (union-SC with capped budgets)

- **Source:** Internal (EXTRACTION_EXPERIMENTS — item-level consensus rewrite, GATED); literature support: self-consistency for extraction via cross-sample fact frequency ([Integrative Decoding, arXiv:2410.01556](https://arxiv.org/abs/2410.01556)); adaptive sample counts ([RASC, arXiv:2408.17017](https://arxiv.org/abs/2408.17017), NAACL 2025).
- **What it is:** Already partially implemented (item-level union consensus, 7 unit tests) but gated after the temp-0.7 reasoning-runaway failure. The re-validation recipe, enriched by the literature: temperature 0.4, explicit `max_tokens` per sample, union keyed by *entity text* not sample-local id (the known collapse trigger), guided JSON (idea 1) so no sample is unparseable, and *adaptive n*: sample 2, add a 3rd only when the first two disagree above a threshold (RASC-style early stop) — applied only to the identification stages, not the whole pipeline.
- **Evidence:** Literature agrees repeatedly-sampled facts are higher-precision and unions raise recall in long-form extraction; internal evidence is direct — union of two RHF runs covered all 12 facts where each single run missed some. The failure evidence is also internal and specific (temp-0.7 runaway; id-keyed union → coref confusion).
- **Fit:** Stage 3/4 via `call_llm_with_self_consistency` (`agents/base.py`). Targets run-to-run variance (Lesson 5) — the same disease as idea 3, attacked by sampling instead of input reshaping. Also un-taxes DomainClassifier (the ~90 s SC escalation loop documented in EXTRACTION_EXPERIMENTS).
- **Experiment:** Already planned in ROADMAP §5.5 — enrich it: after Phase 4, SC-on (temp 0.4, capped, text-keyed) vs SC-off hybrid, slices A+B, 2 repeats; success bar: no collapse, no ladder climbs past 32k, entR variance across repeats halved, mean entR not lower, wall-time ≤1.6× per doc.
- **Effort:** M (most code exists; the delta is config + union keying + adaptive n).

### 7. Cheap wide-harvest union (2-sample singlepass, no consensus machinery)

- **Source:** Internal observation (EXTRACTION_EXPERIMENTS: "the union of the two RHF runs covers all 12 facts"); same principle as sample-union SC but at the one place it's cheapest.
- **What it is:** Run the hybrid front end's single wide call *twice* (temp 0.4 and 0.7), union entities by normalized surface text and relations by (subj,rel-embedding,obj) before seeding the back end. No voting, no confidence blending — pure recall union; the deliberation back end is already the precision filter, which is what distinguishes this from full SC (idea 6) and makes it safe to try first.
- **Evidence:** Internal only, but direct: complementary misses across samples are documented in this codebase on both RHF and singlepass runs. Doubles front-end cost (~20–100 s/doc) — acceptable per the quality-first rule.
- **Fit:** Stage 3 wide-harvest. Targets the residual hybrid entR gap and under-yield docs (a runaway in one sample rarely repeats in the second — cheap insurance where idea 2 is surgical).
- **Experiment:** Post-Phase 4: hybrid-union2 vs hybrid on slices A+B. Success bar: slice-B entR +0.04 or more, precision drop ≤0.03 (back end must absorb the extra candidates), wall +≤120 s/doc. Log how many union-only entities survive to the KG (funnel already counts drops).
- **Effort:** S (hours).

### 8. Direction post-check: value-subject rule + typed spot-check

- **Source:** Internal design (FLIP_ANALYSIS.md post-check §1–2); external motivation: GenRES on why generative RE needs semantic-aware evaluation ([arXiv:2402.10744](https://arxiv.org/abs/2402.10744)).
- **What it is:** Already fully designed in FLIP_ANALYSIS: (1) zero-LLM stage-9 rule — a value-typed node (DATE/TIME/NUMBER by the system's own `classify_value`) may never be the subject of a triple with a non-value object; swap when violated. (2) Bounded LLM spot-check for type-contradicted argument orders (≤1–2 triples/doc). Listed here for completeness because it's literature-adjacent and pre-registered; the flip analysis showed the lever is small (real inversions ≈0.1/doc for hybrid).
- **Evidence:** Internal classification of all 33 flips: ~15% genuine inversions, rhf-concentrated, all matching the value-subject pattern; judge smoke run confirms Class-A flips are inverse-lexicalizations, not errors.
- **Fit:** Stage 9 + Deliberation. Targets Class-B genuine inversions (BORN_ON_DATE/DIED_ON_DATE pattern).
- **Experiment:** As pre-registered in FLIP_ANALYSIS: rhf Class-B flips on docs 0–4 drop 4→≤1 with slice-B metrics unchanged.
- **Effort:** S.

### 9. Two-pass extract-then-verify against source (anti-hallucination glean inversion)

- **Source:** Verification loops in LLM KG construction — surveyed in "LLM-empowered KG construction: a survey" ([arXiv:2510.20345](https://arxiv.org/abs/2510.20345)); mechanism mirrors HippoRAG 2's "recognition memory" filtering idea ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802)) applied at build time.
- **What it is:** The system already has Verification (stage 8), but it verifies holistically. The delta: a *per-triple batched* yes/no check ("is this triple stated or directly implied by THIS sentence?") using the EvidenceLinker's own sentence attributions, at temp 0, guided JSON — one call per doc, list in/list out. Triples failing with high confidence get flagged `unsupported` rather than dropped (governance decides), preserving the never-delete doctrine.
- **Evidence:** Survey-level ("verification loops consistently improve precision") — weak specific evidence; the honest expectation is precision up, recall flat. Internal motivation: entP is 0.79–0.89, so there is headroom, and judge-precision 0.654 suggests some predicted relations are genuinely wrong, not just phrased differently.
- **Fit:** Stage 8 (Verification) with stage-6 evidence as input. Targets relation precision going into Phase B (bad memory facts are worse than missing ones for knowledge-update QA).
- **Experiment:** Post-Phase 4 on slices A+B: entP/relF1 with and without the per-triple check; success bar: relF1@0.6 +0.02 with pairR drop ≤0.01; cost ≤1 call/doc.
- **Effort:** M.

---

## 2. Retrieval / QA ideas (Phase B/C)

### 10. Synonym edges + query-entity-seeded PPR (HippoRAG proper)

- **Source:** HippoRAG ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831), NeurIPS 2024).
- **What it is:** PPR already exists in the QA layer; HippoRAG's measured wins come from two specifics likely missing or partial here: (a) *synonymy edges* — connect entity nodes whose mxbai embeddings exceed a threshold, so PPR mass flows across surface variants the extractor didn't merge (this also makes retrieval robust to deliberately-conservative coref after ideas 4/5); (b) *query-NER seeding* — extract entities from the question with one LLM call and seed the PPR personalization vector exactly on their KG matches, weighted by inverse node frequency (node specificity).
- **Evidence:** Strong, peer-reviewed with ablations: removing node specificity drops MuSiQue R@2 40.9→37.6; HippoRAG under IRCoT gives R@5 +4% MuSiQue, +18% 2Wiki vs ColBERTv2. These are the project's exact Phase C benchmarks.
- **Fit:** QA layer PPR retrieval (`graph_traversal` + `KGVectorStore`). Targets multi-hop retrieval quality for Phase B multi-session questions and Phase C head-to-head vs published HippoRAG numbers.
- **Experiment:** LongMemEval oracle multi-session 20-slice (or Phase C MuSiQue subset): PPR-as-is vs +synonym-edges vs +query-seeding vs both; frozen KGs so it's retrieval-only and cheap. Success bar: answer-evidence recall@k +5pts; judge accuracy +3pts on multi-session.
- **Effort:** M.

### 11. Passage nodes in the graph + PPR over the dual node set (HippoRAG 2)

- **Source:** HippoRAG 2 / "From RAG to Memory" ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802), 2025).
- **What it is:** Add the source chunks/sessions themselves as first-class *passage nodes* linked to the entities extracted from them; run PPR over the mixed phrase+passage graph, seeding both from query-entity matches and from dense query-passage similarity (mxbai); a local-LLM *triple filter* scores which seed triples are actually query-relevant before PPR ("recognition memory"). Retrieval returns passages ranked by PPR mass, so the answer stage reads real text with graph-selected context — fixing the classic KG-QA failure where the graph alone lacks the phrasing needed to answer.
- **Evidence:** Strong: +7 F1 over dense retrievers on associative QA; beats GraphRAG/LightRAG/RAPTOR across factual+associative+sense-making suites while being cheaper at index time. The passage-node move is the single biggest architectural difference from HippoRAG 1.
- **Fit:** QA layer + evidence store (EvidenceLinker already keeps sentence-level provenance — passage nodes are nearly free to materialize). Directly relevant to LongMemEval `_s` haystack condition (B1-4), where retrieving the right *session text* is the game; also the hybrid vector+graph fusion the QA layer already gestures at, made principled.
- **Experiment:** LongMemEval B1-4 prep: on oracle-built KGs, answer with (a) current retrieval, (b) +passage nodes in PPR. Success bar: judge accuracy +5pts on multi-session/temporal slices; retrieval context ≤ same token budget.
- **Effort:** M–L (2–3 days; the provenance links exist, the PPR change is contained).

### 12. Dual-level keyword retrieval wired to community summaries (LightRAG)

- **Source:** LightRAG ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779), EMNLP 2025).
- **What it is:** One LLM call extracts *low-level* keywords (specific entities) and *high-level* keywords (themes/abstractions) from the query. Low-level keywords drive entity-node retrieval (as now); high-level keywords retrieve against the *community summaries* that already exist in this system. Both result sets merge into the answer context. The delta from current state is small: community summaries exist but (per ROADMAP) are consumed by a different path; this gives them a query-conditioned entry point.
- **Evidence:** Peer-reviewed ablation: removing high-level retrieval causes "significant performance decline across nearly all datasets" — the strongest published evidence that theme-level retrieval carries real weight for broad questions.
- **Fit:** AdvancedQAOrchestrator retrieval routing. Targets LongMemEval preference/multi-session questions (broad, aggregative) and Phase C UltraDomain-style global questions where pure entity lookup under-retrieves.
- **Experiment:** LongMemEval oracle: 20-question mixed slice, judge accuracy with/without the high-level branch; success bar: +3pts overall, no regression on single-session (narrow) questions.
- **Effort:** S–M.

### 13. DRIFT-style primer: community-summary-guided sub-query generation

- **Source:** Microsoft GraphRAG DRIFT search — [MS Research blog](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/), [docs](https://microsoft.github.io/graphrag/query/drift_search/).
- **What it is:** Before decomposition, compare the query against top-K community summaries and have the LLM produce (a) a broad primer answer and (b) targeted follow-up sub-queries, which then run through the existing local/PPR path. Delta from current state: query decomposition and active exploration already exist; DRIFT's contribution is *conditioning the decomposition on what the graph actually contains* (the community reports), so sub-queries chase real graph regions instead of hypothetical ones.
- **Evidence:** Microsoft reports large win-rates vs local search on comprehensiveness/diversity; no independent peer-reviewed ablation — moderate evidence, but mechanism is cheap and composable with idea 12.
- **Fit:** AdvancedQAOrchestrator decomposition + active exploration. Targets multi-session LongMemEval questions where the needed sessions aren't surfaced by the literal query terms.
- **Experiment:** Same 20-question multi-session slice: decomposition-as-is vs summary-conditioned decomposition; success bar: evidence-session recall +5pts, ≤2 extra LLM calls/question.
- **Effort:** M.

### 14. Local listwise reranking of retrieved facts/passages before answering

- **Source:** RankZephyr ([arXiv:2312.02724](https://arxiv.org/abs/2312.02724)); FIRST single-token listwise decoding ([arXiv:2406.15657](https://arxiv.org/abs/2406.15657)); EMNLP 2025 empirical survey of LLM rerankers ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.305.pdf)).
- **What it is:** Insert one zero-shot listwise rerank call between retrieval and the answer/debate stage: Nemotron receives the query + the top-20 retrieved items (triples with evidence, or passages) and emits a ranked id list (guided JSON); keep top-k. Listwise beats pointwise for LLM rerankers because relative comparison is in-context. This trims the context the critic/debate agents see, which matters doubly on a thinking model (shorter context → less reasoning ramble at answer time).
- **Evidence:** Consistent literature finding that zero-shot listwise reranking with a capable open model improves retrieval metrics without fine-tuning; magnitude varies by retriever quality (honest: gains shrink when first-stage retrieval is already good).
- **Fit:** QA layer, between PPR/vector retrieval and answer synthesis. Targets LongMemEval `_s` haystack noise (B1-4) — the reportable condition — where first-stage retrieval will be noisiest.
- **Experiment:** B1-4 slice (≤20 questions, `longmemeval_s`): with/without rerank at fixed final context budget. Success bar: judge accuracy +4pts; latency +≤1 call/question.
- **Effort:** S–M.

### 15. Retrieval-sufficiency abstention gate

- **Source:** "Sufficient Context: A New Lens on RAG Systems" (Joren et al., ICLR 2025); AbstentionBench findings on LLM abstention failures; LongMemEval's abstention split (30 `_abs` questions, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)).
- **What it is:** Before answering, one temp-0 guided-JSON call classifies whether the retrieved context is *sufficient* to answer the question (sufficient / insufficient / conflicting). On insufficient → abstain ("I don't have that information"); on conflicting → route into the knowledge-update path (idea 17/18). This is a self-assessment of retrieval, not of the model's parametric knowledge, which is what makes it work for memory QA: the KG either contains the fact or it doesn't.
- **Evidence:** The sufficient-context work shows models answer confidently even with insufficient context and that an explicit sufficiency classifier cuts hallucinated answers substantially; LongMemEval scores abstention explicitly, and commercial assistants do poorly on it. Moderate-to-strong.
- **Fit:** AdvancedQAOrchestrator answer stage / critic. Directly scores on LongMemEval `_abs` ids in every slice (the harness already excludes them from substring scoring — the judge scores them).
- **Experiment:** B1-1 knowledge-update full run already contains `_abs` questions: report abstention accuracy with/without the gate; success bar: abstention accuracy ≥70% with ≤2pts drop on answerable questions (over-abstention is the known failure mode to watch).
- **Effort:** S.

---

## 3. Memory-update & temporal ideas (the thesis; Phase B)

### 16. Bi-temporal fact model: validity intervals on every triple

- **Source:** Zep/Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), 2025); ATOM's dual-time modeling ([arXiv:2510.22590](https://arxiv.org/abs/2510.22590)).
- **What it is:** Extend the `Triple` dataclass with four timestamps: `t_created`/`t_expired` (system time — when the system learned/invalidated it) and `t_valid`/`t_invalid` (world time — when the fact held true). At ingest, the extraction prompt asks for any stated validity cues per fact ("since 2023", "until she moved") — domain-general, no benchmark vocabulary; session/document dates fill `t_created` and default `t_valid`. Nothing is deleted; expiry is metadata (perfectly aligned with the existing never-delete + audit-log governance doctrine).
- **Evidence:** Zep: 71.2% LongMemEval (gpt-4o) vs 60.2% full-context, with temporal-reasoning +17.3pp — the largest published gains on exactly the abilities this project targets. ATOM's dual-time model is the same design validated in a build-time setting.
- **Fit:** Stage 3/4 prompts + `knowledge_graph.py` schema + stage 9. This is the substrate every other memory idea (17, 18, 20) stands on. Prerequisite work is schema-only (no behavior change), so it can be built and unit-tested during any freeze window that allows eval-side work, activated after.
- **Experiment:** B1-2 temporal-reasoning 20-slice (oracle): measure *date survival* (the pre-registered B1-2 gate — session dates → KG) before/after; success bar: ≥90% of evidence-session dates present on the triples that answer, judge accuracy on temporal slice +5pts over the no-timestamps baseline.
- **Effort:** M (schema + prompt fields + plumbing; invalidation logic is idea 17).

### 17. Edge invalidation at ingest (supersede semantics in governance)

- **Source:** Graphiti/Zep edge invalidation ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)); Mem0's LLM conflict resolution marking outdated relations invalid ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)); STALE's implicit-conflict framing ([arXiv:2605.06527](https://arxiv.org/abs/2605.06527)).
- **What it is:** When stage 9 integrates a new triple, retrieve existing triples with the same subject and semantically similar relation (mxbai over relation+object); if candidates exist, one batched LLM call decides per pair: `duplicate | compatible | superseded` — superseded edges get `t_invalid` = new fact's `t_valid` and `t_expired` = now, plus a governance audit entry. This is exactly the system's founding thesis (governed conflict resolution) turned into a concrete, single-call mechanism. Include STALE-style *implicit* conflicts in the prompt ("does the new fact make the old one impossible?"), not just explicit negation.
- **Evidence:** Zep numbers above; Mem0's graph variant credits its temporal-reasoning wins (58.1% vs 21.7% for OpenAI memory on time-sensitive LOCOMO questions) largely to invalidate-don't-delete. STALE (2026) shows even frontier models fail implicit-conflict detection without an explicit mechanism — supporting doing it structurally at ingest rather than hoping the QA model notices.
- **Fit:** Stage 9 / `GovernedKnowledgeGraph` (γ routing already exists — this adds a `supersede` decision type). THE mechanism under test in B1-1 (knowledge-update 78 questions, the first full slice).
- **Experiment:** B1-0 smoke then B1-1: knowledge-update judge accuracy with invalidation on vs off (off = both facts coexist, QA must sort it out). Success bar: +10pts on knowledge-update; zero regressions on single-session slices; every invalidation visible in the audit log (hand-read the 5 smoke graphs).
- **Effort:** M (the retrieval-of-related-edges and governance plumbing exist; the decision call and timestamps are new).

### 18. Deterministic freshness assembly at QA time ("don't ask the LLM to track freshness")

- **Source:** "Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution" ([arXiv:2606.01435](https://arxiv.org/abs/2606.01435), 2026); corroborated by the Supersede paper's memory-update-gap diagnosis ([arXiv:2606.27472](https://arxiv.org/abs/2606.27472)).
- **What it is:** At answer time, group retrieved triples by (subject, relation-cluster); when a group has multiple values with timestamps, *deterministically* select the latest-valid one (plain `max()` over `t_valid`, falling back to `t_created`) and present it as current, with older values shown as "previously: X (until D)". The LLM never has to decide which fact is fresher — the paper shows LLMs are bad at applying freshness rules against their priors even when told explicitly, and that post-retrieval *assembly*, not storage, is the bottleneck.
- **Evidence:** Strong and recent: +28pts over HippoRAG-v2 on MemoryAgentBench FactConsolidation single-hop (78%→94.8% across model tiers), +20pts over best published multi-hop. Supersede paper independently: bounded memory drops knowledge-update accuracy 92%→77% on a frontier model — the gap is real and mechanism-shaped, not scale-shaped.
- **Fit:** QA layer answer-assembly (works even before idea 17 lands, using `t_created` ordering from session dates). Targets B1-1 knowledge-update AND the MemoryAgentBench regression re-run (ROADMAP Phase B #2) — MAB's FactConsolidation is literally the task this recipe was measured on. Cheapest thesis-aligned win available.
- **Experiment:** B1-1: judge accuracy on knowledge-update with (a) raw retrieval, (b) +deterministic assembly, (c) +idea 17 too. Success bar: (b) alone +8pts over (a); (b)+(c) best overall. Zero extra LLM calls.
- **Effort:** S (hours–1 day; it's a sort-and-format function plus relation clustering by embedding).

### 19. Session decomposition + fact-augmented key expansion for the memory index

- **Source:** LongMemEval paper's own design study ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813), ICLR 2025), §memory-design optimizations.
- **What it is:** Two indexing changes the benchmark authors validated: (a) index at *round* granularity (session sliced into user-assistant rounds) instead of whole sessions — better value granularity and token efficiency; (b) *fact-augmented keys*: attach the extracted facts (we have them — they're the KG triples with provenance) as additional retrieval keys pointing back to their source round, so dense search over "facts" finds the round even when the question paraphrases. This is generic memory-index design, not LongMemEval vocabulary; the delta from current state is that EvidenceLinker provenance already links triples→sentences, so (b) is mostly an index-building change over existing data.
- **Evidence:** From the paper itself: recall@5 up to +9.4%, end-to-end QA +5.4pts; fact-expansion noted as most beneficial for knowledge-update and cross-session tracking — our two priority abilities. Caveat honestly: measured by the benchmark's authors on their own benchmark.
- **Fit:** Ingestion wrapper (`AgentGraphMemoryWrapper`) + vector index. Targets B1-4 (`_s` haystack retrieval) and multi-session slices.
- **Experiment:** B1-4 slice: retrieval recall of evidence sessions under (session-level index) vs (round-level + fact keys), frozen KGs. Success bar: evidence recall@5 +6pts, judge accuracy +3pts.
- **Effort:** M.

### 20. Time-aware query expansion (temporal windowing at retrieval)

- **Source:** LongMemEval paper (time-aware indexing/query expansion); Zep's temporal retrieval; Mem0 temporal results (same links as 16/19).
- **What it is:** One temp-0 call extracts any temporal constraint from the question ("last year", "before I moved", "in March") and normalizes it against the question date (already prefixed by the harness); retrieval then *filters or boosts* candidates whose `t_valid`/`t_created` fall in the window before semantic ranking. Requires idea 16's timestamps (or minimally session dates on provenance).
- **Evidence:** LongMemEval authors: temporal-awareness worth 7–11% on temporal reasoning — their single largest lever. Zep's biggest relative gain was also temporal (+17.3pp).
- **Fit:** QA layer retrieval pre-filter. Targets B1-2 temporal-reasoning (133 questions — the largest ability split). Domain-general: dates come from the corpus and question, no benchmark vocabulary.
- **Experiment:** B1-2 20-slice: judge accuracy ± windowing on the same KGs. Success bar: +7pts on temporal slice, no regression elsewhere (windowing must fail open — if no temporal cue is detected, retrieval is unchanged).
- **Effort:** S–M.

### 21. Sleep-time consolidation pass (offline KnowledgeOrganizer re-run)

- **Source:** Letta sleep-time compute ([blog](https://www.letta.com/blog/sleep-time-compute/)); A-MEM's memory-evolution principle ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110), NeurIPS 2025).
- **What it is:** A scheduled offline job (between ingest sessions, no user-facing latency) that re-reads the most recently touched subgraph and: re-runs coref/dedupe across session boundaries (cross-session aliases that per-session ingest can't see), refreshes entity descriptions in light of new facts (A-MEM's "new memories trigger updates to old memories' contextual representations"), recomputes affected community summaries, and runs the idea-17 conflict check retroactively across sessions. Delta from current state: stage 9 and community summaries exist — this is *re-invoking them cross-session on a schedule* rather than only at ingest.
- **Evidence:** Mechanism-level (Letta reports quality/latency benefits, no rigorous public benchmark; A-MEM shows gains across six models on memory QA but with its own memory format). Moderate evidence; low risk because it reuses audited components.
- **Fit:** Stage 9 + community summaries, cross-session. Targets multi-session linking (B1-3 gate: "cross-session linking sanity") — per-session ingest provably cannot merge an entity first seen in session 2 with its alias in session 40 unless something looks across sessions.
- **Experiment:** B1-3 multi-session slice: ingest all sessions, then answer (a) immediately vs (b) after one consolidation pass. Success bar: cross-session entity duplicates (same person, two nodes) reduced ≥50% by hand-count on 5 graphs; judge accuracy +4pts on multi-session.
- **Effort:** M.

---

## 4. Phase D self-improvement ideas

### 22. GEPA-style reflective prompt evolution with the DocRED/LongMemEval harness as metric

- **Source:** GEPA: Reflective Prompt Evolution Can Outperform RL ([arXiv:2507.19457](https://arxiv.org/abs/2507.19457), ICLR 2026 oral; [DSPy implementation](https://dspy.ai/api/optimizers/GEPA/overview/)).
- **What it is:** Tier-1 self-improvement, upgraded: instead of blind config sweeps, GEPA mutates stage prompts by *reflecting in natural language on execution traces* (our per-doc funnel logs, drop reasons, judge verdicts are unusually rich feedback text) and keeps a Pareto frontier of prompt candidates. Nemotron plays both roles (actor + reflector). The harness score (entR/pairR/relF1 or LongMemEval judge acc) is the metric; funnel logs are the feedback string.
- **Evidence:** Strong and directly relevant: beats GRPO by ~6–19pts with up to 35× fewer rollouts, beats MIPROv2 by >10pts — sample efficiency is exactly what a local-GPU, minutes-per-doc harness needs. Not yet demonstrated on KG extraction specifically.
- **Fit:** Phase D Tier 1. Optimizes the wide-harvest and coref prompts first (the two highest-leverage prompts per the matrix arc). Anti-overfit guard maps directly onto existing doctrine: optimize on slice A-class docs, verdict on held-out + a non-DocRED corpus.
- **Experiment:** 10 evolution steps × 5-doc evals (≈50 doc-runs, one overnight) on the wide-harvest prompt; success bar: held-out slice entR/pairR ≥ frozen hybrid v2 +0.03, AND no regression on a synthetic non-DocRED doc set (bias guard).
- **Effort:** L (3+ days incl. wrapper), but Tier 1 was already committed roadmap.

### 23. Lessons-as-memory: retrievable extraction exemplars per domain (ExpeL)

- **Source:** ExpeL: LLM Agents Are Experiential Learners ([arXiv:2308.10144](https://arxiv.org/abs/2308.10144), AAAI 2024).
- **What it is:** Tier-2 enrichment: convert the hand-read lessons (LESSONS.md failure modes, judge-identified errors, deliberation reject rationales) into a small store of natural-language *insights* + concrete success/failure exemplars, keyed by the DomainClassifier's discovered domain. At extraction time, retrieve the top-2 insights for the current domain and inject into the stage prompt ("in narrative-biography documents, dates are entities, not attributes"). Insights are distilled by comparing successful vs failed doc-runs (ExpeL's compare-and-extract recipe) — the system literally eats its own benchmark lessons.
- **Evidence:** ExpeL shows consistent gains across tasks without fine-tuning; evidence is agent-benchmark-based, not extraction-based — moderate transfer. Dogfooding note: this is precisely what the graphify/mem0 tooling around this project already does for code sessions.
- **Fit:** Phase D Tier 2 (explicitly on the roadmap); touches stage prompts only, keeps domain-adaptivity (insights are per-discovered-domain, learned from the system's own runs, no benchmark vocabulary hardcoded).
- **Experiment:** Seed the store from LESSONS.md + judge output on Phase 4 caches; A/B on a fresh 10-doc slice: insights-on vs off. Success bar: entR/relF1 +0.03 held-out; insights must be verifiably domain-conditional (log which fired).
- **Effort:** M.

### 24. DPO-LoRA on deliberation/governance accept-reject pairs (enriching planned Tier 3)

- **Source:** Iterative DPO for extraction with recall-ranked preference pairs (practice documented in e.g. [ChemoTimelines 2025, arXiv:2512.04518](https://arxiv.org/pdf/2512.04518) and ICLR 2025 self-training work); Supersede's GRPO result on memory updates ([arXiv:2606.27472](https://arxiv.org/abs/2606.27472)).
- **What it is:** Already planned; the literature adds three concrete design choices: (a) *on-policy pairs* — generate candidates with the current Nemotron, don't import pairs from other models (measurably better); (b) chosen/rejected by *harness signal*, not human labels: chosen = extraction sample whose triples the judge scorer matches to gold/deliberation-approved set with higher recall; rejected = lowest; (c) a second pair source is free: governance approve vs reject decisions with their rationales, collected passively during Phase B/C. Supersede shows GRPO on a supersession environment nearly doubled a small model's held-out supersession accuracy — evidence that the knowledge-update behavior specifically is trainable with exactly the reward this project can compute.
- **Evidence:** Moderate-to-strong for the recipe pattern; no public result for DPO-LoRA on a 30B FP8 Nemotron (local trainability of the LoRA needs a feasibility spike first — FP8 base + LoRA is nonstandard).
- **Fit:** Phase D Tier 3, as planned. Data collection can start *now* at zero cost (log pairs during Phase B); training is the later, riskier half.
- **Experiment (data half only):** During B1-1, log (sample, judge-recall) tuples and governance decisions; success bar for the collection phase: ≥500 clean pairs with ≥0.2 recall spread between chosen/rejected.
- **Effort:** S for collection now; L for training later.

### 25. Judge-scorer-driven config search (Tier 1 floor, before/alongside GEPA)

- **Source:** Internal (`score_docred.py --judge`, smoke-validated; Optuna TPE precedent already in `scripts/tune/`); the pattern is Cognee's Dreamify (ARCHITECTURE_COMPARISON §2).
- **What it is:** The minimal self-improvement loop that needs zero new research: TPE/random search over the small discrete config space (temperatures per stage, glean trigger threshold from idea 2, coref merge threshold from idea 4, SC n) with the offline judge scorer on cached extractions as objective where possible (cheap) and 5-doc re-extractions where not. Delta from current state: both halves exist (Optuna harness, judge scorer); they've never been connected.
- **Evidence:** Internal + the general AutoML pattern; Cognee's Dreamify demonstrates the same 6-parameter loop works for a comparable pipeline. Weak published numbers; high implementation certainty.
- **Fit:** Phase D Tier 1 floor. Also the natural first consumer of Phase 4's 40-doc caches.
- **Experiment:** 20-trial TPE over 4 knobs on slice-A-class docs, verdict on held-out. Success bar: any config beating frozen hybrid v2 held-out relF1 by ≥0.02 without entR loss; if none found, that's a publishable-in-MATRIX_REPORT null result bounding the config headroom.
- **Effort:** M.

---

## 5. Out of scope but notable (one line each)

- **GLiNER/GLiREL/spaCy NER** — fixed general-purpose extractors; explicitly rejected by project constraint.
- **AutoGraph-R1** ([arXiv:2510.15339](https://arxiv.org/pdf/2510.15339)) — end-to-end RL for KG construction; weeks of training infra, violates days-scale.
- **Fine-tuned rerankers (RankZephyr training recipe, FIRST fine-tune)** — training beyond planned DPO-LoRA; zero-shot variant kept as idea 14.
- **LazyGraphRAG** — noun-phrase co-occurrence graphs replace LLM extraction; undercuts the domain-adaptive schema thesis (its *deferred-summarization* spirit already exists as lazy ingest).
- **KAG (OpenSPG)** ([arXiv:2409.13731](https://arxiv.org/abs/2409.13731)) — strong 2Wiki numbers (+33.5% F1 rel.) but its logical-form engine + schema framework is a platform migration, not a days-scale delta; mutual-indexing idea is captured by idea 11.
- **StructRAG** ([arXiv:2410.08815](https://arxiv.org/abs/2410.08815)) — inference-time re-structurization into tables/graphs per query; interesting but router training + format library is weeks.
- **Docs2KG / AutoKG** — heterogeneous-document unified KG / lightweight keyword KG; neither adds beyond ideas already listed for a text-corpus system.
- **MemGPT/Letta OS-style context paging** ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)) — orthogonal to a KG memory; sleep-time consolidation kept as idea 21.
- **Mem0 ADD-only pipeline rewrite** — their v3 abandoned UPDATE/DELETE for ADD-only + retrieval-side resolution; notable as convergent evidence for idea 18's assembly-side thesis.
- **Budget forcing / s1-style "Wait"-token control** ([arXiv:2501.19393](https://arxiv.org/abs/2501.19393)) — needs decode-loop control vLLM doesn't expose cleanly for this build; the truncation-retry ladder + ideas 1/3 cover the practical need.
- **Think-on-Graph 3.0 / agentic graph reasoning frameworks** — heavy multi-agent retrieval stacks; this system already has its own agentic QA layer to benchmark first.

---

## 6. Top 5 ranked recommendations (effort × expected gain × evidence)

1. **#18 Deterministic freshness assembly (S effort).** Strongest evidence-per-hour in this doc: +28pts on the exact task family (FactConsolidation/knowledge-update) that is this system's thesis and first Phase B slice; zero LLM calls; QA-side only, so it can be built and unit-tested without touching frozen `multi_agent_kg/` extraction paths. Do it before B1-1 launches.
2. **#17 + #16 Edge invalidation on a bi-temporal substrate (M).** The thesis mechanism itself, validated by Zep (71.2% LongMemEval, temporal +17.3pp) and Mem0-graph; slots into governance the system already has. #18 covers QA-time; this makes the graph itself update-aware and is what B1-1 exists to measure.
3. **#1 Guided JSON structured decoding (S–M).** Retires an entire failure class (empty/malformed output after reasoning burn) that caused the project's founding incident and still taxes every run via ladder rescues; makes SC (#6), verification (#9), and every QA classifier call cheaper and safer. Low ceiling on quality metrics, very high floor on robustness.
4. **#10/#11 HippoRAG-style PPR upgrades — synonym edges, query seeding, passage nodes (M–L).** The QA layer is about to be benchmarked for the first time; these are the highest-evidence retrieval deltas available (peer-reviewed ablations on MuSiQue/2Wiki, the project's own Phase C targets), and they compound with conservative coref (#4/#5) rather than fighting it. Start with #10 (smaller), take #11 into B1-4.
5. **#2 Targeted low-yield re-glean (S–M).** The one extraction idea aimed at a *specific, localized, still-open* gap (reasoning-runaway docs like doc 33 — the only systematic hybrid deficit left after v5). Cheap, trigger-gated, GraphRAG-precedented; a clean one-loop experiment on the existing DocRED harness the moment Phase 4 finishes.

Near-misses: #7 (union wide-harvest) is the fastest extraction experiment if #2 disappoints; #4/#5 (coref gating) if Phase 4's n=40 shows the coref over-merge gap is still material; #22 (GEPA) is the standout Phase D bet once Phase B numbers exist to optimize against.
