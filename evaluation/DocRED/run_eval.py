"""Re-DocRED extraction runner — Phase 0/1/2 of evaluation/DocRED/PLAN.md.

Runs the KG pipeline over Re-DocRED dev docs and checkpoints one JSON per doc
(predicted graph + gold reference + per-call LLM stats) so that:
  - a crash loses at most one document,
  - scoring (score_docred.py) is fully offline — never re-extract to re-score.

Usage (Phase 0 smoke):
  python -u evaluation/DocRED/run_eval.py --max-docs 1 \
      --save-kg-dir evaluation/results/docred_kg_cache \
      --output evaluation/results/docred_run.json
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

# ── per-call instrumentation (reset per doc) ─────────────────────────────────
CALLS = []
_orig_chat = oc.chat_completion


def _timed(messages, *a, **k):
    t = time.time()
    out = _orig_chat(messages, *a, **k)
    CALLS.append({"wall_s": round(time.time() - t, 2), "chars": len(out or "")})
    return out


oc.chat_completion = _timed

from multi_agent_kg.core.config import LLMConfig  # noqa: E402
from multi_agent_kg.agents.base import ModelTier  # noqa: E402


def detokenize(sents):
    """Join Re-DocRED token lists into readable prose (light touch)."""
    text = " ".join(" ".join(s) for s in sents)
    for a, b in ((" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"),
                 ("( ", "("), (" )", ")"), (" '", "'"), (" !", "!"), (" ?", "?")):
        text = text.replace(a, b)
    return text


def gold_reference(doc):
    """Gold clusters + triples in the scorer's shape (kept beside predictions)."""
    clusters = [
        {"mentions": sorted({m["name"] for m in vs}), "type": vs[0]["type"]}
        for vs in doc["vertexSet"]
    ]
    triples = [
        {"h": t["h"], "t": t["t"], "r": t["r"], "evidence": t.get("evidence", [])}
        for t in doc["labels"]
    ]
    return {"clusters": clusters, "triples": triples}


def run_rhf(text, model, self_consistency=False, extraction_mode="deliberative"):
    from multi_agent_kg.core import DeliberativeOrchestrator
    from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
    from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph

    gkg = GovernedKnowledgeGraph(governance_mode="permissive")
    orch = DeliberativeOrchestrator(
        llm_config=LLMConfig(model=model),
        knowledge_graph=KnowledgeGraph(),
        governed_kg=gkg,
        quality_threshold=0.35,
        max_refinement_iterations=1,
        enable_self_consistency=self_consistency,
        enable_open_world=True,
        enable_cross_document=False,
        model_tiers={t: model for t in ModelTier},
        extraction_mode=extraction_mode,
    )
    error = None
    try:
        orch.process_corpus([{"text": text, "metadata": {"source": "docred"}}])
    except Exception as exc:  # preserve the partial KG — never lose the doc
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    gkg = orch.governed_kg
    # Dump the entity's full label set (every mention surface coref merged in),
    # not just the canonical name. Coref renames "FIBT World Championships" to a
    # canonical id; the gold-matchable surfaces live in labels. Omitting them
    # made the scorer under-count hybrid/rhf recall vs singlepass (which keeps
    # raw surfaces as its name) — a measurement artifact, not a real gap.
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
    return entities, triples, error


def run_singlepass(text, model):
    """GraphRAG-style one-call extraction + gleaning (same as scripts/extraction_experiment)."""
    prompt = (
        "You are extracting a knowledge graph from text.\n\n"
        "Extract EVERY significant entity and EVERY relationship in ONE pass, including dates, "
        "years, quantities and numeric values as entities when a fact connects to them. Use the "
        "exact surface form for entity names. Prefer specific relations.\n\n"
        f"TEXT:\n{text}\n\n"
        'Return ONLY JSON:\n{"entities":[{"name":"...","type":"..."}],\n'
        ' "relationships":[{"source":"...","relation":"...","target":"..."}]}'
    )
    res = oc.chat_completion_json(
        messages=[{"role": "system", "content": "You are a precise knowledge-graph extractor. Return ONLY valid JSON."},
                  {"role": "user", "content": prompt}],
        model=model, temperature=0.1, max_tokens=4096)
    ents = res.get("entities", []) if isinstance(res, dict) else []
    rels = res.get("relationships", []) if isinstance(res, dict) else []
    entities = [{"id": e.get("name", ""), "type": str(e.get("type", "?")), "name": e.get("name", "")}
                for e in ents if isinstance(e, dict) and e.get("name")]
    triples = [{"subject": str(r.get("source")), "relation": str(r.get("relation")),
                "object": str(r.get("target")), "confidence": 0.7}
               for r in rels if isinstance(r, dict) and r.get("source") and r.get("target")]
    return entities, triples, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default=os.path.join(os.path.dirname(__file__), "data", "dev_revised.json"))
    ap.add_argument("--max-docs", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--strategy", choices=["rhf", "singlepass", "hybrid", "spgov"], default="rhf")
    ap.add_argument("--sc", action="store_true", help="self-consistency (GATED — see EXTRACTION_EXPERIMENTS.md)")
    ap.add_argument("--save-kg-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = os.environ.get("LLM_DEFAULT_MODEL", "nvidia/nemotron-3-nano")
    os.makedirs(args.save_kg_dir, exist_ok=True)
    with open(args.data_file) as f:
        docs = json.load(f)[args.offset: args.offset + args.max_docs]

    print(f"[DOCRED] strategy={args.strategy} sc={args.sc} model={model} "
          f"docs={args.offset}..{args.offset + len(docs) - 1} "
          f"NEMOTRON_THINKING={oc.NEMOTRON_THINKING}", flush=True)

    summary = []
    for i, doc in enumerate(docs):
        idx = args.offset + i
        cache_path = os.path.join(args.save_kg_dir, f"doc_{idx}_{args.strategy}.json")
        if os.path.exists(cache_path):
            print(f"[DOCRED] doc {idx} cached — skipping", flush=True)
            with open(cache_path) as f:
                summary.append(json.load(f)["counts"] | {"idx": idx, "cached": True})
            continue

        text = detokenize(doc["sents"])
        print(f"\n[DOCRED] doc {idx}: '{doc['title']}' "
              f"({len(text)} chars, gold: {len(doc['vertexSet'])} ents / {len(doc['labels'])} triples)",
              flush=True)
        CALLS.clear()
        t0 = time.time()
        if args.strategy == "rhf":
            entities, triples, error = run_rhf(text, model, self_consistency=args.sc)
        elif args.strategy == "hybrid":
            entities, triples, error = run_rhf(text, model, self_consistency=args.sc,
                                               extraction_mode="wide")
        elif args.strategy == "spgov":
            # GB-9: governed singlepass — wide harvest becomes the triples,
            # RHF/evidence/deliberation skipped, verify + governed commit kept.
            entities, triples, error = run_rhf(text, model, self_consistency=args.sc,
                                               extraction_mode="governed_singlepass")
        else:
            entities, triples, error = run_singlepass(text, model)
        wall = round(time.time() - t0, 1)

        counts = {
            "wall_s": wall,
            "llm_calls": len(CALLS),
            "empty_calls": sum(1 for c in CALLS if c["chars"] == 0),
            "pred_entities": len(entities),
            "pred_triples": len(triples),
            "gold_entities": len(doc["vertexSet"]),
            "gold_triples": len(doc["labels"]),
            "error": error,
        }
        record = {
            "idx": idx,
            "title": doc["title"],
            "strategy": args.strategy,
            "model": model,
            "counts": counts,
            "entities": entities,
            "triples": triples,
            "gold": gold_reference(doc),
            "text": text,
            "calls": CALLS[:],
        }
        with open(cache_path, "w") as f:      # checkpoint IMMEDIATELY per doc
            json.dump(record, f, indent=2)
        print(f"[DOCRED] doc {idx} done in {wall}s: {len(entities)} ents / {len(triples)} triples "
              f"({counts['llm_calls']} calls, {counts['empty_calls']} empty)"
              + (f" ERROR={error}" if error else ""), flush=True)
        summary.append(counts | {"idx": idx})

    with open(args.output, "w") as f:
        json.dump({"strategy": args.strategy, "sc": args.sc, "model": model, "docs": summary}, f, indent=2)
    print(f"\n[DOCRED] RUN_COMPLETE — {len(summary)} docs -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
