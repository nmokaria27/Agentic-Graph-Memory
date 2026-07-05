"""Offline scorer for Re-DocRED runs — reads run_eval.py's per-doc cache.

Layered so we see WHERE triples die (evaluation/DocRED/PLAN.md):
  L1 entity:   predicted surface ↔ gold mention cluster (P/R/F1)
  L2 pair:     predicted (head, tail) hits a gold pair, either direction
  L3 relation: predicted relation name ≈ gold relation label
               (exact after normalization, else embedding cosine ≥ threshold)

Direction flips are counted separately — a lesson category, not silent noise.

Usage:
  python evaluation/DocRED/score_docred.py \
      --kg-dir evaluation/results/docred_kg_cache --strategy rhf \
      [--no-embed] [--output evaluation/results/docred_scores.json]
"""
import argparse
import glob
import json
import math
import os
import re
import string
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def norm(s):
    s = str(s).replace("_", " ").lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def _load_rel_info():
    path = os.path.join(os.path.dirname(__file__), "data", "rel_info.json")
    with open(path) as f:
        return json.load(f)


# ── entity ↔ cluster matching ────────────────────────────────────────────────
def match_entity(pred_name, clusters_norm):
    """Best gold cluster for a predicted surface: exact mention first, then containment."""
    p = norm(pred_name)
    if not p:
        return None
    for ci, mentions in enumerate(clusters_norm):
        if p in mentions:
            return ci
    if len(p) >= 4:
        for ci, mentions in enumerate(clusters_norm):
            for m in mentions:
                if len(m) >= 4 and (p in m or m in p):
                    return ci
    return None


# ── relation similarity ──────────────────────────────────────────────────────
class RelSim:
    def __init__(self, use_embed=True):
        self.use_embed = use_embed
        self._vecs = {}

    def _vec(self, phrase):
        if phrase not in self._vecs:
            from multi_agent_kg.llm.openai_client import get_embeddings
            self._vecs[phrase] = get_embeddings([phrase])[0]
        return self._vecs[phrase]

    def prefetch(self, phrases):
        todo = [p for p in dict.fromkeys(phrases) if p not in self._vecs]
        if not todo or not self.use_embed:
            return
        from multi_agent_kg.llm.openai_client import get_embeddings
        for phrase, vec in zip(todo, get_embeddings(todo)):
            self._vecs[phrase] = vec

    def sim(self, a, b):
        a, b = norm(a), norm(b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if not self.use_embed:  # token-Jaccard fallback
            ta, tb = set(a.split()), set(b.split())
            return len(ta & tb) / len(ta | tb)
        va, vb = self._vec(a), self._vec(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        return dot / (na * nb) if na and nb else 0.0


def score_doc(rec, rel_info, rel_sim, thresholds):
    gold = rec["gold"]
    clusters_norm = [{norm(m) for m in c["mentions"]} for c in gold["clusters"]]

    # L1 — entities. Match against the entity's full surface set (name + every
    # coref-merged label), not just the canonical name — the gold-matchable
    # mention forms live in labels. Falls back to name/id for caches without a
    # labels key (e.g. singlepass), so those score identically to before.
    def _surfaces(e):
        surfs = [e.get("name") or e.get("id")]
        surfs.extend(e.get("labels", []) or [])
        return [s for s in surfs if s]

    def _match_any(surfaces):
        for s in surfaces:
            ci = match_entity(s, clusters_norm)
            if ci is not None:
                return ci
        return None

    ent_map = {e["id"]: _match_any(_surfaces(e)) for e in rec["entities"]}
    matched_preds = sum(1 for v in ent_map.values() if v is not None)
    matched_clusters = len({v for v in ent_map.values() if v is not None})
    n_pred, n_gold = len(ent_map), len(clusters_norm)
    ent = {
        "precision": matched_preds / n_pred if n_pred else 0.0,
        "recall": matched_clusters / n_gold if n_gold else 0.0,
    }

    # L2 — pairs (+ L3 relation on top of matched pairs)
    gold_pairs = {}
    for t in gold["triples"]:
        gold_pairs.setdefault((t["h"], t["t"]), []).append(rel_info.get(t["r"], t["r"]))

    pred_mapped = []  # (ch, ct, relation) for triples with both endpoints matched
    for t in rec["triples"]:
        ch = ent_map.get(t["subject"], match_entity(t["subject"], clusters_norm))
        ct = ent_map.get(t["object"], match_entity(t["object"], clusters_norm))
        if ch is not None and ct is not None and ch != ct:
            pred_mapped.append((ch, ct, t["relation"]))

    hit_pairs, flipped_pairs = set(), set()
    for ch, ct, _ in pred_mapped:
        if (ch, ct) in gold_pairs:
            hit_pairs.add((ch, ct))
        elif (ct, ch) in gold_pairs:
            flipped_pairs.add((ct, ch))
    pair = {
        "gold_pairs": len(gold_pairs),
        "hit": len(hit_pairs),
        "flipped_only": len(flipped_pairs - hit_pairs),
        "recall_either_direction": (len(hit_pairs | flipped_pairs) / len(gold_pairs)) if gold_pairs else 0.0,
        "pred_pairs": len({(a, b) for a, b, _ in pred_mapped}),
    }

    # L3 — relation correctness on correctly-directed pairs
    rel_sim.prefetch([norm(r) for _, _, r in pred_mapped]
                     + [norm(x) for rels in gold_pairs.values() for x in rels])
    gold_total = sum(len(v) for v in gold_pairs.values())
    rel_scores = {}
    for th in thresholds:
        correct_gold = 0
        for (h, t), rels in gold_pairs.items():
            preds_here = [r for ch, ct, r in pred_mapped if (ch, ct) == (h, t)]
            for grel in rels:
                if any(rel_sim.sim(pr, grel) >= th for pr in preds_here):
                    correct_gold += 1
        correct_pred = 0
        for ch, ct, pr in pred_mapped:
            rels = gold_pairs.get((ch, ct), [])
            if any(rel_sim.sim(pr, grel) >= th for grel in rels):
                correct_pred += 1
        p = correct_pred / len(pred_mapped) if pred_mapped else 0.0
        r = correct_gold / gold_total if gold_total else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rel_scores[th] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    # unmatched gold clusters by TYPE — the "where recall dies" diagnostic
    missed_types = {}
    hit_cluster_ids = {v for v in ent_map.values() if v is not None}
    for ci, c in enumerate(gold["clusters"]):
        if ci not in hit_cluster_ids:
            missed_types[c["type"]] = missed_types.get(c["type"], 0) + 1

    return {"idx": rec["idx"], "title": rec["title"],
            "entity": {k: round(v, 3) for k, v in ent.items()},
            "pair": pair, "relation": rel_scores,
            "missed_gold_entity_types": missed_types,
            "counts": rec["counts"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", required=True)
    ap.add_argument("--strategy", default="rhf")
    ap.add_argument("--thresholds", default="0.6,0.7,0.8")
    ap.add_argument("--no-embed", action="store_true", help="token-Jaccard instead of embeddings")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rel_info = _load_rel_info()
    rel_sim = RelSim(use_embed=not args.no_embed)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    files = sorted(glob.glob(os.path.join(args.kg_dir, f"doc_*_{args.strategy}.json")))
    if not files:
        sys.exit(f"no cached docs for strategy={args.strategy} in {args.kg_dir}")

    results = []
    for path in files:
        with open(path) as f:
            rec = json.load(f)
        s = score_doc(rec, rel_info, rel_sim, thresholds)
        results.append(s)
        mid = thresholds[len(thresholds) // 2]
        print(f"doc {s['idx']:>3} '{s['title'][:38]:<38}' "
              f"entP={s['entity']['precision']:.2f} entR={s['entity']['recall']:.2f} "
              f"pairR={s['pair']['recall_either_direction']:.2f} "
              f"flip={s['pair']['flipped_only']} "
              f"relF1@{mid}={s['relation'][mid]['f1']:.2f} "
              f"missed={s['missed_gold_entity_types']}")

    # micro aggregate (weighted by doc gold sizes via simple mean of ratios for now)
    n = len(results)
    agg = {
        "docs": n,
        "entity_precision": round(sum(r["entity"]["precision"] for r in results) / n, 3),
        "entity_recall": round(sum(r["entity"]["recall"] for r in results) / n, 3),
        "pair_recall": round(sum(r["pair"]["recall_either_direction"] for r in results) / n, 3),
        "direction_flips": sum(r["pair"]["flipped_only"] for r in results),
        "relation": {th: {
            "precision": round(sum(r["relation"][th]["precision"] for r in results) / n, 3),
            "recall": round(sum(r["relation"][th]["recall"] for r in results) / n, 3),
            "f1": round(sum(r["relation"][th]["f1"] for r in results) / n, 3),
        } for th in thresholds},
    }
    print("\nAGGREGATE:", json.dumps(agg, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"aggregate": agg, "per_doc": results}, f, indent=2)
        print(f"scores -> {args.output}")


if __name__ == "__main__":
    main()
