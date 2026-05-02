#!/usr/bin/env python3
"""Summarize relation-extraction funnel diagnostics from build outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


NUMERIC_FIELDS = [
    "segments_processed",
    "entities_seen",
    "relations_found",
    "head_bindings",
    "tail_triples",
    "pairwise_pairs_considered",
    "pairwise_positive_predictions",
    "pairwise_triples_added",
    "gleaned_triples_added",
    "invalid_self_refs_filtered",
    "post_alignment_triples",
    "post_dedupe_triples",
    "final_triples",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _extract_summaries(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    direct = data.get("relation_funnel_summary")
    if isinstance(direct, dict):
        yield direct

    for section in ("governed", "ungoverned"):
        section_payload = data.get(section)
        if not isinstance(section_payload, dict):
            continue
        summary = section_payload.get("relation_funnel_summary")
        if isinstance(summary, dict):
            tagged = dict(summary)
            tagged["_section"] = section
            yield tagged
        aggregate = section_payload.get("aggregate")
        if isinstance(aggregate, dict):
            summary = aggregate.get("relation_funnel_summary")
            if isinstance(summary, dict):
                tagged = dict(summary)
                tagged["_section"] = section
                yield tagged


def _merge(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {field: 0 for field in NUMERIC_FIELDS}
    merged["documents_with_diagnostics"] = 0
    merged["docs_with_zero_final_triples"] = []
    merged["docs_with_stage1_relations_but_no_final_triples"] = []
    merged["per_doc"] = []

    for summary in summaries:
        for field in NUMERIC_FIELDS:
            value = summary.get(field, 0)
            if isinstance(value, (int, float)):
                merged[field] += value
        docs = summary.get("documents_with_diagnostics", 0)
        if isinstance(docs, int):
            merged["documents_with_diagnostics"] += docs
        for field in ("docs_with_zero_final_triples", "docs_with_stage1_relations_but_no_final_triples"):
            values = summary.get(field, [])
            if isinstance(values, list):
                merged[field].extend(str(value) for value in values)
        per_doc = summary.get("per_doc", [])
        if isinstance(per_doc, list):
            merged["per_doc"].extend(row for row in per_doc if isinstance(row, dict))

    if merged["relations_found"]:
        merged["tail_triples_per_stage1_relation"] = round(
            merged["tail_triples"] / merged["relations_found"], 4
        )
    else:
        merged["tail_triples_per_stage1_relation"] = 0.0
    if merged["post_alignment_triples"]:
        merged["dedupe_retention"] = round(
            merged["post_dedupe_triples"] / merged["post_alignment_triples"], 4
        )
    else:
        merged["dedupe_retention"] = 0.0
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Stats or comparison JSON files to summarize")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    by_file: Dict[str, Any] = {}
    all_summaries: List[Dict[str, Any]] = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        summaries = list(_extract_summaries(_load_json(path)))
        if not summaries:
            by_file[str(path)] = {"error": "no relation_funnel_summary found"}
            continue
        merged = _merge(summaries)
        by_file[str(path)] = merged
        all_summaries.extend(summaries)

    payload = {
        "files": by_file,
        "combined": _merge(all_summaries),
    }

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
