"""Scorer entity-matching: canonical name misses gold, but a merged label hits.

Guards the measurement fix — coref renames entities to canonical ids, so the
gold-matchable surface forms live in `labels`. The scorer must match name UNION
labels, and stay backward-compatible with caches that have no labels key.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation", "DocRED"))

from score_docred import RelSim, score_doc  # noqa: E402

_REL_INFO = {"P17": "country", "P131": "located in"}


def _rec(entities):
    return {
        "idx": 0, "title": "t", "counts": {},
        "gold": {
            "clusters": [
                {"mentions": ["FIBT World Championships", "the championships"], "type": "MISC"},
                {"mentions": ["Mediaș"], "type": "LOC"},
            ],
            "triples": [],
        },
        "entities": entities,
        "triples": [],
    }


def test_canonical_name_misses_but_label_hits() -> None:
    # name is a mangled canonical id (no gold match); the merged label is the
    # real surface and must recover the hit.
    rec = _rec([
        {"id": "fibt_world_championsh", "name": "fibt_world_championsh",
         "labels": ["FIBT World Championships"]},
    ])
    s = score_doc(rec, _REL_INFO, RelSim(use_embed=False), [0.6])
    assert s["entity"]["recall"] == 0.5  # 1 of 2 gold clusters hit via label


def test_backward_compatible_without_labels_key() -> None:
    # Caches predating the fix (singlepass) have no labels — must still match on name.
    rec = _rec([{"id": "medias", "name": "Mediaș"}])
    s = score_doc(rec, _REL_INFO, RelSim(use_embed=False), [0.6])
    assert s["entity"]["recall"] == 0.5


def test_empty_labels_list_falls_back_to_name() -> None:
    rec = _rec([{"id": "medias", "name": "Mediaș", "labels": []}])
    s = score_doc(rec, _REL_INFO, RelSim(use_embed=False), [0.6])
    assert s["entity"]["recall"] == 0.5
