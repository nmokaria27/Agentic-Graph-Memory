"""Coref-collapse repro harness (LESSONS.md Lesson 1).

Re-runs the RHF pipeline on one Re-DocRED doc with EVERY LLM call's full
prompt + response dumped to a JSONL, so we can see exactly which stage
turns N extracted entities into 1. No pipeline code changes needed.

Usage:
  python -u evaluation/DocRED/repro_coref.py --doc 3 \
      --io-log evaluation/results/coref_repro_io.jsonl
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import multi_agent_kg.llm.openai_client as oc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--doc", type=int, default=3)
ap.add_argument("--data-file", default=os.path.join(os.path.dirname(__file__), "data", "dev_revised.json"))
ap.add_argument("--io-log", default="evaluation/results/coref_repro_io.jsonl")
args = ap.parse_args()

_log = open(args.io_log, "w")
_orig = oc.chat_completion
_n = [0]


def _logged(messages, *a, **k):
    out = _orig(messages, *a, **k)
    _n[0] += 1
    _log.write(json.dumps({
        "call": _n[0],
        "system": next((m["content"] for m in messages if m.get("role") == "system"), "")[:400],
        "prompt": next((m["content"] for m in messages if m.get("role") == "user"), ""),
        "response": out,
    }) + "\n")
    _log.flush()
    print(f"  [IO] call {_n[0]} logged ({len(out or '')} chars out)", flush=True)
    return out


oc.chat_completion = _logged

from run_eval import detokenize, run_rhf  # noqa: E402  (same dir)

with open(args.data_file) as f:
    doc = json.load(f)[args.doc]
text = detokenize(doc["sents"])
model = os.environ.get("LLM_DEFAULT_MODEL", "nvidia/nemotron-3-nano")
print(f"[REPRO] doc {args.doc} '{doc['title']}' gold={len(doc['vertexSet'])} ents", flush=True)

entities, triples, error = run_rhf(text, model)
print(f"\n[REPRO] FINAL: {len(entities)} entities / {len(triples)} triples err={error}", flush=True)
for e in entities:
    print(f"  ent: {e['id']} ({e['type']})", flush=True)
print(f"[REPRO] IO log -> {args.io_log}", flush=True)
print("[REPRO] DONE", flush=True)
