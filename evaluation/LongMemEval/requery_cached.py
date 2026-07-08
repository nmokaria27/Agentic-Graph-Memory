"""Re-run ONLY the QA step for cached LongMemEval questions.

Extraction is the expensive part (~45-60 min/question local); the QA query is
~30 s. When a QA-layer bug crashes the answer but the KG built fine (e.g. the
_community_context AttributeError, EXP-FRESHNESS-E2E local leg), this reloads
each question's final governed-KG checkpoint, rebuilds the QA system on the
FIXED code, re-asks the question, and updates the cache JSON in place
(hypothesis + a requery marker). Rescore with score_longmemeval.py afterwards.

Usage:
    python evaluation/LongMemEval/requery_cached.py \
        --cache-dir evaluation/results/lme_kg_cache_qwen3_freshness \
        --qids 6a1eabeb 830ce83f 945e3d21
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_wrapper_class():
    path = os.path.join(os.path.dirname(__file__), "..", "Memory-Agent-Bench",
                        "agent_graph_memory_adapter.py")
    spec = importlib.util.spec_from_file_location("agent_graph_memory_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AgentGraphMemoryWrapper


def best_kg_snapshot(ckpt_context_dir: str):
    """The per-document snapshots accumulate; take the one with most triples."""
    best, best_n = None, -1
    for f in glob.glob(os.path.join(ckpt_context_dir, "doc_*", "governed_kg_latest.json")):
        try:
            d = json.load(open(f))
            n = len((d.get("knowledge_graph") or d).get("triples", []))
        except Exception:
            continue
        if n > best_n:
            best, best_n = d, n
    return best, best_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--qids", nargs="+", required=True)
    args = ap.parse_args()

    from multi_agent_kg.core import GovernedKnowledgeGraph

    Wrapper = load_wrapper_class()

    for qid in args.qids:
        cache_path = os.path.join(args.cache_dir, f"{qid}.json")
        record = json.load(open(cache_path))
        idx = record["idx"]
        ckpt_dir = os.path.join(args.cache_dir, "ckpt", f"context_{idx}")
        snapshot, n_triples = best_kg_snapshot(ckpt_dir)
        if snapshot is None:
            print(f"[REQUERY] {qid}: NO KG SNAPSHOT under {ckpt_dir} — skipping")
            continue

        gkg = GovernedKnowledgeGraph.from_dict(snapshot)
        print(f"[REQUERY] {qid} (idx {idx}): restored KG with "
              f"{len(gkg.entities)} entities / {len(gkg.kg.triples)} triples")

        wrapper = Wrapper(model=record["model"], answer_format="longmemeval",
                          extraction_mode=record.get("extraction_mode", "wide"))
        wrapper._context_id = idx
        wrapper._governed_kg = gkg
        wrapper._qa_system = wrapper._build_qa_system(gkg)
        wrapper._qa_system.governed_kg = gkg

        question_text = f"The current date is {record['question_date']}. {record['question']}"
        t0 = time.time()
        resp = wrapper.send_message(question_text, memorizing=False,
                                    query_id=0, context_id=idx)
        hyp = resp.get("output", "")
        dt = round(time.time() - t0, 1)

        record["hypothesis_pre_requery"] = record.get("hypothesis", "")
        record["hypothesis"] = hyp
        record["requery"] = {"reason": "qa_fix__community_context", "query_s": dt}
        with open(cache_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[REQUERY] {qid}: {dt}s -> hyp={hyp[:120]!r}")
        print(f"          gold: {record['answer'][:80]!r}\n")

    print("REQUERY_DONE")


if __name__ == "__main__":
    main()
