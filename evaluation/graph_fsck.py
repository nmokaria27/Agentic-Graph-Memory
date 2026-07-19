#!/usr/bin/env python3
"""graph fsck (PLAYBOOK §3.4): structural invariant checker for dumped KGs.

Reads a KG artifact (per-doc eval dump, kg_export, or governed_kg checkpoint)
and asserts the invariants that past incidents violated:
  I1  every triple endpoint resolves to an entity (id or surface)
  I2  no triple is both superseded and active
  I3  active triples carry at least one provenance ref (when provenance exists
      anywhere in the artifact — legacy dumps without provenance are skipped)
  I4  no entity id appears twice
Prints violations; exit code 1 if any. Never auto-fixes.

Usage: python evaluation/graph_fsck.py <artifact.json> [more.json ...]
       python evaluation/graph_fsck.py cache_dir/            (checks every *.json)
"""
import json
import re
import sys
from pathlib import Path


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def check(path: Path) -> int:
    try:
        d = json.load(open(path))
    except Exception as exc:
        print(f"{path}: UNREADABLE ({exc})")
        return 1
    kg = d.get("knowledge_graph") or d
    ents = kg.get("entities")
    trs = kg.get("triples")
    if ents is None or trs is None:
        return 0  # not a KG artifact — skip silently (scores, stats, etc.)

    if isinstance(ents, dict):
        ids = set(ents.keys())
        rows = list(ents.values())
    else:
        ids = {e.get("id") for e in ents if isinstance(e, dict)}
        rows = [e for e in ents if isinstance(e, dict)]

    surfaces = set()
    for e in rows:
        for s in [e.get("name"), e.get("text")] + list(e.get("labels") or []):
            if s:
                surfaces.add(_norm(s))
    known = {_norm(i) for i in ids if i} | surfaces

    bad = 0
    # I4 duplicate ids (list-form only; dict keys are unique by construction)
    if not isinstance(ents, dict):
        seen = set()
        for e in rows:
            eid = e.get("id")
            if eid in seen:
                print(f"{path}: I4 duplicate entity id {eid!r}")
                bad += 1
            seen.add(eid)

    any_prov = any(
        isinstance(t, dict) and (t.get("metadata") or {}).get("provenance")
        for t in trs
    )
    for i, t in enumerate(trs):
        if not isinstance(t, dict):
            print(f"{path}: triple[{i}] not a dict")
            bad += 1
            continue
        for end_key, id_key in (("subject", "subject_id"), ("object", "object_id")):
            cand = [_norm(t.get(id_key)), _norm(t.get(end_key))]
            if not any(c and c in known for c in cand):
                print(f"{path}: I1 dangling {end_key} {t.get(end_key)!r} "
                      f"({t.get('relation')})")
                bad += 1
        meta = t.get("metadata") or {}
        if t.get("superseded") and (t.get("active") or meta.get("active")):
            print(f"{path}: I2 triple[{i}] superseded AND active")
            bad += 1
        if any_prov and not t.get("superseded") and not meta.get("provenance"):
            print(f"{path}: I3 active triple[{i}] has no provenance "
                  f"({t.get('subject')} {t.get('relation')} {t.get('object')})")
            bad += 1
    return bad


def main():
    targets = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        targets.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    if not targets:
        print(__doc__)
        sys.exit(2)
    total = sum(check(p) for p in targets)
    print(f"fsck: {len(targets)} artifact(s), {total} violation(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
