"""Direction-flip analysis — offline mining of existing DocRED caches.

ROADMAP §5.3: for every predicted pair that hits a gold pair ONLY in the
reverse direction (score_docred.py's `flipped_only` category), record which
gold relation was flipped, what relation name the system predicted, and the
entity types involved — so we can see WHICH relations invert (BORN_IN,
LOCATED_IN, ...) and design a targeted direction post-check.

Zero LLM / zero embedding calls; reuses score_docred's own norm/match_entity
so a "flip" here is exactly what the scorer counts. Safe during Phase 4.

Usage:
  python evaluation/DocRED/analyze_flips.py \
      --kg-dirs evaluation/results/docred_kg_cache evaluation/results/docred_kg_cache_heldout \
      --strategies hybrid rhf singlepass \
      --output evaluation/results/flip_analysis.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_docred import norm, match_entity, _load_rel_info  # noqa: E402


def flips_in_doc(rec, rel_info):
    """Replicates score_doc's L2 mapping, keeping the flipped triples."""
    gold = rec["gold"]
    clusters_norm = [{norm(m) for m in c["mentions"]} for c in gold["clusters"]]
    cluster_types = [c["type"] for c in gold["clusters"]]

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

    gold_pairs = {}
    for t in gold["triples"]:
        gold_pairs.setdefault((t["h"], t["t"]), []).append(rel_info.get(t["r"], t["r"]))

    flips, hits = [], set()
    for t in rec["triples"]:
        ch = ent_map.get(t["subject"], match_entity(t["subject"], clusters_norm))
        ct = ent_map.get(t["object"], match_entity(t["object"], clusters_norm))
        if ch is None or ct is None or ch == ct:
            continue
        if (ch, ct) in gold_pairs:
            hits.add((ch, ct))
        elif (ct, ch) in gold_pairs:
            flips.append({
                "doc_idx": rec["idx"],
                "title": rec.get("title", ""),
                "pred": {"subject": t["subject"], "relation": t["relation"],
                         "object": t["object"]},
                "gold_relations": gold_pairs[(ct, ch)],
                "gold_head_type": cluster_types[ct],
                "gold_tail_type": cluster_types[ch],
                "pair": (ct, ch),
            })
    # a pair that ALSO hit in the correct direction is not flipped_only
    flips = [f for f in flips if f["pair"] not in hits]
    for f in flips:
        f.pop("pair")
    return flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dirs", nargs="+", required=True)
    ap.add_argument("--strategies", nargs="+",
                    default=["hybrid", "rhf", "singlepass"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--examples-per-relation", type=int, default=3)
    args = ap.parse_args()

    rel_info = _load_rel_info()
    by_strategy = {}
    for strat in args.strategies:
        all_flips, n_docs = [], 0
        for d in args.kg_dirs:
            for path in sorted(glob.glob(os.path.join(d, f"doc_*_{strat}.json"))):
                with open(path) as f:
                    rec = json.load(f)
                n_docs += 1
                all_flips.extend(flips_in_doc(rec, rel_info))

        gold_rel_counts = Counter(g for f in all_flips for g in f["gold_relations"])
        pred_rel_counts = Counter(norm(f["pred"]["relation"]) for f in all_flips)
        type_pairs = Counter(f"{f['gold_head_type']}->{f['gold_tail_type']}"
                             for f in all_flips)
        examples = defaultdict(list)
        for f in all_flips:
            for g in f["gold_relations"]:
                if len(examples[g]) < args.examples_per_relation:
                    p = f["pred"]
                    examples[g].append(
                        f"doc {f['doc_idx']} '{f['title'][:28]}': "
                        f"pred [{p['subject']} --{p['relation']}--> {p['object']}] "
                        f"but gold direction is reversed")

        by_strategy[strat] = {
            "docs_scanned": n_docs,
            "total_flipped_only": len(all_flips),
            "flips_per_doc": round(len(all_flips) / n_docs, 2) if n_docs else 0.0,
            "by_gold_relation": dict(gold_rel_counts.most_common()),
            "by_predicted_relation": dict(pred_rel_counts.most_common(25)),
            "by_gold_type_pair": dict(type_pairs.most_common(15)),
            "examples": dict(examples),
        }

        print(f"\n=== {strat}: {len(all_flips)} flipped-only pairs "
              f"across {n_docs} docs ===")
        for rel, cnt in gold_rel_counts.most_common(15):
            print(f"  {cnt:>3}  {rel}")
        print("  top gold type pairs:",
              dict(type_pairs.most_common(6)))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"kg_dirs": args.kg_dirs, "strategies": by_strategy}, f, indent=2)
        print(f"\nflip analysis -> {args.output}")


if __name__ == "__main__":
    main()
