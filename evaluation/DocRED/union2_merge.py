"""EXP-UNION2-SINGLEPASS — merge two independent singlepass samples into a
synthetic cache dir, scoreable by the existing score_docred.py (no new
scoring logic; reuses the tracked scorer for an apples-to-apples comparison
against sample A alone / sample B alone).

Union rule (no consensus machinery, per idea #7):
  - entities: union by id (surface text), keep first-seen record
  - triples: union by (subject, relation, object) exact tuple, keep first-seen
No confidence blending, no voting — pure recall union; the point is to
measure whether complementary misses across two independent draws recover
more of the gold pairs than either draw alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def merge_doc(doc_a: dict, doc_b: dict) -> dict:
    ents_by_id = {}
    for e in doc_a["entities"] + doc_b["entities"]:
        key = e.get("id", e.get("name", e.get("text")))
        ents_by_id.setdefault(key, e)

    trips_by_key = {}
    for t in doc_a["triples"] + doc_b["triples"]:
        key = (t.get("subject"), t.get("relation"), t.get("object"))
        trips_by_key.setdefault(key, t)

    merged = dict(doc_a)  # carries idx/title/gold/model
    merged["strategy"] = "singlepass"
    merged["entities"] = list(ents_by_id.values())
    merged["triples"] = list(trips_by_key.values())
    merged["counts"] = {
        "wall_s": doc_a["counts"]["wall_s"] + doc_b["counts"]["wall_s"],
        "llm_calls": doc_a["counts"]["llm_calls"] + doc_b["counts"]["llm_calls"],
        "empty_calls": doc_a["counts"]["empty_calls"] + doc_b["counts"]["empty_calls"],
        "pred_entities": len(merged["entities"]),
        "pred_triples": len(merged["triples"]),
        "gold_entities": doc_a["counts"]["gold_entities"],
        "gold_triples": doc_a["counts"]["gold_triples"],
        "error": None,
        "union_only_entities": len(ents_by_id) - len(
            {e.get("id", e.get("name", e.get("text"))) for e in doc_a["entities"]}
        ),
        "union_only_triples": len(trips_by_key) - len(
            {(t.get("subject"), t.get("relation"), t.get("object")) for t in doc_a["triples"]}
        ),
    }
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files_a = sorted(glob.glob(os.path.join(args.dir_a, "doc_*_singlepass.json")))
    total_union_only_ents = total_union_only_trips = 0
    n = 0
    for fa in files_a:
        base = os.path.basename(fa)
        fb = os.path.join(args.dir_b, base)
        if not os.path.exists(fb):
            print(f"  SKIP {base}: no matching sample-B file")
            continue
        doc_a = json.load(open(fa))
        doc_b = json.load(open(fb))
        merged = merge_doc(doc_a, doc_b)
        with open(os.path.join(args.out_dir, base), "w") as f:
            json.dump(merged, f, indent=2)
        n += 1
        total_union_only_ents += merged["counts"]["union_only_entities"]
        total_union_only_trips += merged["counts"]["union_only_triples"]
        print(f"  {base}: A={len(doc_a['entities'])}e/{len(doc_a['triples'])}t "
              f"B={len(doc_b['entities'])}e/{len(doc_b['triples'])}t -> "
              f"union={len(merged['entities'])}e/{len(merged['triples'])}t "
              f"(+{merged['counts']['union_only_entities']}e/"
              f"+{merged['counts']['union_only_triples']}t new from B)")

    print(f"\nMerged {n} docs -> {args.out_dir}")
    print(f"Total union-only (B-contributed) entities: {total_union_only_ents}, "
          f"triples: {total_union_only_trips}")


if __name__ == "__main__":
    main()
