"""EXP-DEDUP-GUARD (GB-8) — portable prototype of the organizer dedup guard.

Built freeze-safe (outside multi_agent_kg/) while EXP-ROBUST-VALIDATE holds the
freeze. `apply_merge_groups_guarded` is written to slot verbatim into
`KnowledgeOrganizer._deduplicate_entities` (knowledge_organizer.py ~L382-401),
replacing the unbounded delete-on-merge loop that collapsed Qwen3 hybrid slice A
(doc_0: 32 extracted -> 2 kept, 27/29 triple endpoints dangling).

Three domain-general guards (no dataset vocabulary):
  1. type-compatibility gate  — a PERSON cannot merge into a DATE
  2. merge-group size cap      — reject groups swallowing >abs_cap or >rel_cap of the doc
  3. merge-into-aliases        — copy merged surface forms onto the canonical (no loss)

Run directly to execute the fault-injection self-tests:
    python evaluation/DocRED/dedup_guard_prototype.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _entity_id(e: Dict[str, Any]) -> str:
    return e.get("id", e.get("text", ""))


def _entity_type(e: Dict[str, Any]) -> str:
    return (e.get("type") or "").strip().upper()


def apply_merge_groups_guarded(
    remaining: List[Dict[str, Any]],
    merge_groups: List[Any],
    abs_cap: int = 8,
    rel_cap: float = 0.5,
) -> Tuple[List[Dict[str, Any]], int, int, int, List[Tuple[str, str]]]:
    """Apply LLM merge groups with type/size guards and alias preservation.

    Returns (new_remaining, applied_merges, skipped_groups, malformed_groups,
    alias_pairs) where alias_pairs is [(alias_id, canonical_id), ...] for
    shared_memory.register_entity_alias, matching the existing call shape.

    A merge is applied to entity E only when:
      - the group is a well-formed dict with canonical_id + merge_ids
      - E's type matches the canonical entity's type (empty type = permissive)
      - the group's type-eligible merge_ids do not exceed abs_cap or rel_cap
    The canonical entity absorbs each merged entity's text/labels as aliases.
    """
    by_id: Dict[str, Dict[str, Any]] = {_entity_id(e): e for e in remaining}
    doc_size = len(remaining)
    removed: set = set()
    applied = skipped = malformed = 0
    alias_pairs: List[Tuple[str, str]] = []

    for group in merge_groups:
        # dict-shape contract (kept from the existing malformed-group guard)
        if not isinstance(group, dict):
            malformed += 1
            continue
        canonical_id = group.get("canonical_id")
        merge_ids = group.get("merge_ids", [])
        if not (canonical_id and merge_ids):
            continue

        canonical = by_id.get(canonical_id)
        if canonical is None:
            # canonical not in the working set (already merged / unknown) -> skip
            skipped += 1
            continue
        canon_type = _entity_type(canonical)

        # Guard 1: type-compatibility gate (empty canonical type = permissive)
        eligible = []
        for mid in merge_ids:
            if mid == canonical_id or mid in removed:
                continue
            target = by_id.get(mid)
            if target is None:
                continue
            t_type = _entity_type(target)
            if not canon_type or not t_type or t_type == canon_type:
                eligible.append(mid)

        if not eligible:
            continue

        # Guard 2: merge-group size cap (absolute OR relative to document size)
        rel_limit = max(1, int(doc_size * rel_cap))
        if len(eligible) > abs_cap or len(eligible) > rel_limit:
            skipped += 1
            continue

        # Guard 3: merge-into-aliases — canonical absorbs merged surface forms
        labels = canonical.setdefault("labels", [])
        if canonical.get("text") and canonical["text"] not in labels:
            labels.append(canonical["text"])
        for mid in eligible:
            target = by_id[mid]
            for surf in [target.get("text")] + list(target.get("labels", [])):
                if surf and surf not in labels:
                    labels.append(surf)
            alias_pairs.append((mid, canonical_id))
            removed.add(mid)
            applied += 1

    new_remaining = [e for e in remaining if _entity_id(e) not in removed]
    return new_remaining, applied, skipped, malformed, alias_pairs


# --------------------------------------------------------------------------- #
# Fault-injection self-tests (the pre-registered success bar (a)).
# On port these become tests/test_dedup_guard.py against knowledge_organizer.
# --------------------------------------------------------------------------- #
def _ent(i, text, typ):
    return {"id": i, "text": text, "type": typ}


def _run_selftests() -> None:
    # --- Case 1: the doc_0 catastrophe — aggressive cross-type mega-merge ---
    ents = [
        _ent("wilfried_schneider", "Wilfried Schneider", "PERSON"),
        _ent("turin", "Turin", "LOC"),
        _ent("german", "German", "NATIONALITY"),
        _ent("british_columbia", "British Columbia", "LOC"),
        _ent("skeleton_racer", "skeleton racer", "OCCUPATION"),
        _ent("melissa_hollingsworth", "Melissa Hollingsworth", "PERSON"),
        _ent("1997", "1997", "DATE"),
        _ent("2002", "2002", "DATE"),
        _ent("13_march_1963", "13 March 1963", "DATE"),
        _ent("july_2012", "July 2012", "DATE"),
    ]
    # LLM tries to merge everything into a single DATE node (mixed types, huge group)
    bad_group = [{
        "canonical_id": "13_march_1963",
        "canonical_name": "13 March 1963",
        "merge_ids": [e["id"] for e in ents if e["id"] != "13_march_1963"],
    }]
    kept, applied, skipped, malformed, aliases = apply_merge_groups_guarded(ents, bad_group)
    kept_ids = {e["id"] for e in kept}
    # The guard's contract: NO cross-type deletion. Every entity whose type differs
    # from the canonical (DATE) must survive. (Same-type dates may still collapse —
    # that is the LLM's semantic call, out of scope for the deletion-safety guard.)
    non_dates = [e["id"] for e in ents if _entity_type(e) != "DATE"]
    survivors = [i for i in non_dates if i in kept_ids]
    assert survivors == non_dates, f"cross-type entities deleted: {set(non_dates) - kept_ids}"
    assert len(kept) >= 7, f"catastrophic collapse not prevented: kept {len(kept)}/{len(ents)}"
    print(f"  [1] cross-type mega-merge BLOCKED: kept {len(kept)}/{len(ents)}; "
          f"all {len(non_dates)} non-DATE entities survived (buggy code kept 2)")

    # --- Case 2: legitimate same-type duplicate still merges + preserves alias ---
    ents2 = [
        _ent("melissa_hollingsworth", "Melissa Hollingsworth", "PERSON"),
        _ent("m_hollingsworth", "M. Hollingsworth", "PERSON"),
        _ent("turin", "Turin", "LOC"),
    ]
    good_group = [{
        "canonical_id": "melissa_hollingsworth",
        "canonical_name": "Melissa Hollingsworth",
        "merge_ids": ["m_hollingsworth"],
    }]
    kept2, applied2, skipped2, _, aliases2 = apply_merge_groups_guarded(ents2, good_group)
    assert len(kept2) == 2 and applied2 == 1, (len(kept2), applied2)
    canon = next(e for e in kept2 if e["id"] == "melissa_hollingsworth")
    assert "M. Hollingsworth" in canon["labels"], canon["labels"]
    assert aliases2 == [("m_hollingsworth", "melissa_hollingsworth")], aliases2
    print(f"  [2] legit same-type merge APPLIED + alias preserved: labels={canon['labels']}")

    # --- Case 3: type-mismatch inside an otherwise-small group is filtered ---
    ents3 = [
        _ent("apple_inc", "Apple Inc.", "ORG"),
        _ent("apple", "Apple", "ORG"),
        _ent("tim_cook", "Tim Cook", "PERSON"),  # wrong type, must NOT merge
    ]
    mixed_group = [{
        "canonical_id": "apple_inc",
        "canonical_name": "Apple Inc.",
        "merge_ids": ["apple", "tim_cook"],
    }]
    kept3, applied3, _, _, _ = apply_merge_groups_guarded(ents3, mixed_group)
    ids3 = {e["id"] for e in kept3}
    assert "tim_cook" in ids3 and "apple" not in ids3, ids3
    assert applied3 == 1, applied3
    print(f"  [3] type-mismatch filtered: kept={sorted(ids3)} (PERSON survived)")

    # --- Case 4: malformed (non-dict) group counted, not crashed ---
    kept4, _, _, malformed4, _ = apply_merge_groups_guarded(
        ents3, ["Apple Inc.", {"canonical_id": "apple_inc", "merge_ids": ["apple"]}]
    )
    assert malformed4 == 1, malformed4
    print(f"  [4] malformed non-dict group counted (malformed={malformed4}), no crash")

    print("ALL DEDUP-GUARD SELF-TESTS PASSED")


if __name__ == "__main__":
    _run_selftests()
