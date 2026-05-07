#!/usr/bin/env python3
"""Merge over-fragmented open-world org-chart domains in a governed KG file."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph


def compact_domains(graph: GovernedKnowledgeGraph) -> dict:
    original_domains = list(graph.org_chart.domains)
    compacted = []
    added = 0
    merged = 0

    for candidate in original_domains:
        exact = next((domain for domain in compacted if domain.domain_id == candidate.domain_id), None)
        target = exact or DeliberativeOrchestrator._schema_merge_target(candidate, compacted)
        if target is None:
            compacted.append(candidate)
            added += 1
            continue

        target.entity_ids.update(candidate.entity_ids)
        DeliberativeOrchestrator._merge_domain_schema(target, candidate)
        merged += 1

    graph.org_chart.domains = compacted
    graph.org_chart._entity_domain_map_cache = None
    graph.org_chart.refresh_cross_domain_relations(graph.kg)
    stats = graph.get_stats()
    graph.set_bootstrap_assignment_stats(
        {
            **stats.get("bootstrap_assignment_stats", {}),
            "domain_compaction": {
                "original_domains": len(original_domains),
                "compacted_domains": len(compacted),
                "added_roots": added,
                "merged_domains": merged,
            },
        }
    )
    return graph.get_stats()["bootstrap_assignment_stats"]["domain_compaction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backup-original",
        action="store_true",
        help="Copy the input to <input>.pre_compaction.bak before writing output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    processed_ids = data.get("_processed_doc_ids")
    graph = GovernedKnowledgeGraph.from_dict(data)

    compaction = compact_domains(graph)
    payload = graph.to_dict()
    payload["stats"] = graph.get_stats()
    if processed_ids is not None:
        payload["_processed_doc_ids"] = processed_ids

    if args.backup_original:
        backup = input_path.with_suffix(input_path.suffix + ".pre_compaction.bak")
        if not backup.exists():
            shutil.copy2(input_path, backup)

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **compaction}, indent=2))


if __name__ == "__main__":
    main()
