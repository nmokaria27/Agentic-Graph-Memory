"""LongMemEval runner — Phase B #1 of ROADMAP.md.

Runs the full KG pipeline + QA layer over LongMemEval questions and checkpoints
one JSON per question (hypothesis answer + KG dump + gold + per-call LLM stats)
so that:
  - a crash loses at most one question,
  - scoring (score_longmemeval.py) is fully offline — never re-run to re-score.

Mirrors the DocRED harness conventions (evaluation/DocRED/run_eval.py):
per-item checkpoints, cache-skip on rerun, instrumented LLM calls,
--max-questions/--offset slicing.

Each question is an independent context: haystack sessions are formatted as
dated conversation transcripts and ingested through AgentGraphMemoryWrapper
(the same adapter MemoryAgentBench uses — 9-agent pipeline + orphan relink +
AdvancedQAOrchestrator with hybrid retrieval), then the question is asked once.

Usage (smoke slice — 5 knowledge-update questions on the oracle split):
  NEMOTRON_THINKING=on python -u evaluation/LongMemEval/run_eval.py \
      --question-type knowledge-update --max-questions 5 \
      --save-kg-dir evaluation/results/lme_kg_cache_smoke \
      --output evaluation/results/lme_smoke.json

Ability order (PLAN.md): knowledge-update FIRST, then temporal-reasoning,
multi-session, single-session-*, abstention (scored via _abs question ids).
"""
import argparse
import json
import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import multi_agent_kg.llm.openai_client as oc  # noqa: E402

# ── per-call instrumentation (reset per question) ────────────────────────────
CALLS = []
_orig_chat = oc.chat_completion


def _timed(messages, *a, **k):
    t = time.time()
    out = _orig_chat(messages, *a, **k)
    CALLS.append({"wall_s": round(time.time() - t, 2), "chars": len(out or "")})
    return out


oc.chat_completion = _timed

QUESTION_TYPES = [
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]


def load_wrapper_class():
    """Import AgentGraphMemoryWrapper from the hyphenated Memory-Agent-Bench dir.

    Loaded AFTER the chat_completion patch above so every pipeline/QA call is
    instrumented (same ordering trick as the DocRED runner).
    """
    import importlib.util

    path = os.path.join(_ROOT, "evaluation", "Memory-Agent-Bench",
                        "agent_graph_memory_adapter.py")
    spec = importlib.util.spec_from_file_location("agent_graph_memory_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AgentGraphMemoryWrapper


def format_session(date, turns):
    """One haystack session -> a dated transcript document.

    The date header is load-bearing: knowledge-update and temporal-reasoning
    questions are only answerable if session order/recency survives ingestion.
    """
    lines = [f"Conversation between the user and the assistant on {date}:"]
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        content = str(t.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def dump_kg(wrapper):
    """Entity/triple dump in the DocRED cache shape (labels included — the
    coref-merged surfaces are the gold-matchable ones; see matrix v4)."""
    gkg = wrapper._governed_kg
    if gkg is None:
        return [], []
    entities = [
        {"id": eid, "type": str(getattr(e, "entity_type", "?")),
         "name": str(getattr(e, "name", eid)),
         "labels": [str(l) for l in (getattr(e, "labels", None) or [])]}
        for eid, e in gkg.entities.items()
    ]
    triples = [
        {"subject": t.subject, "relation": t.relation, "object": t.object,
         "confidence": float(getattr(t, "confidence", 0.0) or 0.0)}
        for t in gkg.triples
    ]
    return entities, triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file",
                    default=os.path.join(os.path.dirname(__file__), "data",
                                         "longmemeval_oracle.json"),
                    help="oracle (evidence-only) or longmemeval_s (full haystack)")
    ap.add_argument("--question-type", choices=QUESTION_TYPES, default=None,
                    help="filter to one ability; default = all")
    ap.add_argument("--max-questions", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0,
                    help="offset INTO the filtered question list")
    ap.add_argument("--save-kg-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--extraction-mode", choices=["deliberative", "wide"],
                    default="wide",
                    help="wide = frozen hybrid v2 (matrix v5 decision)")
    ap.add_argument("--no-date-prefix", action="store_true",
                    help="omit 'The current date is ...' from the question")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the selected questions and exit (no LLM calls)")
    args = ap.parse_args()

    model = os.environ.get("LLM_DEFAULT_MODEL", "nvidia/nemotron-3-nano")
    with open(args.data_file) as f:
        data = json.load(f)
    pool = [q for q in data
            if args.question_type is None or q["question_type"] == args.question_type]
    questions = pool[args.offset: args.offset + args.max_questions]

    print(f"[LME] type={args.question_type or 'ALL'} model={model} "
          f"mode={args.extraction_mode} data={os.path.basename(args.data_file)} "
          f"questions={args.offset}..{args.offset + len(questions) - 1} of {len(pool)} "
          f"NEMOTRON_THINKING={oc.NEMOTRON_THINKING}", flush=True)

    if args.dry_run:
        for i, q in enumerate(questions):
            n_turns = sum(len(s) for s in q["haystack_sessions"])
            chars = sum(len(str(t.get("content", "")))
                        for s in q["haystack_sessions"] for t in s)
            print(f"  [{args.offset + i}] {q['question_id']} ({q['question_type']}"
                  f"{', ABSTENTION' if '_abs' in str(q['question_id']) else ''}): "
                  f"{q['question'][:80]!r} — {len(q['haystack_sessions'])} sessions / "
                  f"{n_turns} turns / {chars} chars", flush=True)
        print("[LME] DRY_RUN_COMPLETE", flush=True)
        return

    os.makedirs(args.save_kg_dir, exist_ok=True)
    Wrapper = load_wrapper_class()

    summary = []
    for i, q in enumerate(questions):
        idx = args.offset + i
        qid = str(q["question_id"])
        cache_path = os.path.join(args.save_kg_dir, f"{qid}.json")
        if os.path.exists(cache_path):
            print(f"[LME] q {idx} ({qid}) cached — skipping", flush=True)
            with open(cache_path) as f:
                summary.append(json.load(f)["counts"] | {"idx": idx, "cached": True})
            continue

        sessions = [format_session(d, s)
                    for d, s in zip(q["haystack_dates"], q["haystack_sessions"])]
        total_chars = sum(len(s) for s in sessions)
        print(f"\n[LME] q {idx} ({qid}, {q['question_type']}): {q['question'][:90]!r} "
              f"— {len(sessions)} sessions / {total_chars} chars", flush=True)

        CALLS.clear()
        t0 = time.time()
        error = None
        hypothesis = ""
        build_time = query_time = 0.0
        entities, triples = [], []
        # Fresh wrapper per question — contexts are independent by construction.
        # checkpoint_dir gives stage-level resume WITHIN a question (matters for
        # the big longmemeval_s haystacks); the per-question cache file above is
        # the question-level resume.
        wrapper = Wrapper(model=model, answer_format="longmemeval",
                          extraction_mode=args.extraction_mode,
                          checkpoint_dir=os.path.join(args.save_kg_dir, "ckpt"))
        try:
            for s in sessions:
                wrapper.send_message(s, memorizing=True, context_id=idx)
            question_text = q["question"] if args.no_date_prefix else (
                f"The current date is {q['question_date']}. {q['question']}")
            resp = wrapper.send_message(question_text, memorizing=False,
                                        query_id=0, context_id=idx)
            hypothesis = resp.get("output", "")
            build_time = round(float(resp.get("memory_construction_time", 0.0)), 1)
            query_time = round(float(resp.get("query_time_len", 0.0)), 1)
        except Exception as exc:  # preserve the partial record — never lose the question
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        try:
            entities, triples = dump_kg(wrapper)
        except Exception as exc:
            print(f"  WARNING: KG dump failed ({exc})", flush=True)
        wall = round(time.time() - t0, 1)

        # A pipeline that spent many LLM calls but committed nothing has failed
        # SILENTLY (e.g. a stage exception marked the doc failed and stage 9 never
        # committed — EXP-2 q0, 186 calls -> 0 entities, error None). Surface it:
        # the scorer and verdicts must see this as an error, not a valid empty KG.
        if error is None and not entities and len(CALLS) > 20:
            error = (f"SUSPECTED_SILENT_PIPELINE_FAILURE: 0 entities committed "
                     f"after {len(CALLS)} LLM calls (check failed_documents in log)")
            print(f"  WARNING: {error}", flush=True)

        counts = {
            "wall_s": wall,
            "build_s": build_time,
            "query_s": query_time,
            "llm_calls": len(CALLS),
            "empty_calls": sum(1 for c in CALLS if c["chars"] == 0),
            "n_sessions": len(sessions),
            "context_chars": total_chars,
            "kg_entities": len(entities),
            "kg_triples": len(triples),
            "error": error,
        }
        record = {
            "idx": idx,
            "question_id": qid,
            "question_type": q["question_type"],
            "question": q["question"],
            "question_date": q["question_date"],
            "answer": q["answer"],
            "answer_session_ids": q.get("answer_session_ids", []),
            "haystack_session_ids": q.get("haystack_session_ids", []),
            "haystack_dates": q.get("haystack_dates", []),
            "extraction_mode": args.extraction_mode,
            "model": model,
            "hypothesis": hypothesis,
            "counts": counts,
            "entities": entities,
            "triples": triples,
            "calls": CALLS[:],
        }
        with open(cache_path, "w") as f:      # checkpoint IMMEDIATELY per question
            json.dump(record, f, indent=2)
        print(f"[LME] q {idx} done in {wall}s (build {build_time}s / query {query_time}s): "
              f"{len(entities)} ents / {len(triples)} triples, "
              f"hyp={hypothesis[:100]!r}"
              + (f" ERROR={error}" if error else ""), flush=True)
        summary.append(counts | {"idx": idx, "question_id": qid})

    with open(args.output, "w") as f:
        json.dump({"question_type": args.question_type, "model": model,
                   "extraction_mode": args.extraction_mode,
                   "data_file": args.data_file, "questions": summary}, f, indent=2)
    print(f"\n[LME] RUN_COMPLETE — {len(summary)} questions -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
