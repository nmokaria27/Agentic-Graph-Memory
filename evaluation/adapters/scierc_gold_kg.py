"""
Build a larger KG artifact directly from SciERC gold annotations.

This avoids the slow extraction pipeline by importing the annotated
entities and relations as a graph in the repo's native KG format.
It also builds a deterministic heuristic org chart so larger ablation
runs do not depend on the expensive LLM-based DomainBuilder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from evaluation.adapters.scierc_adapter import SciERCAdapter
from multi_agent_kg.core.domain_experts import Domain, OrgChart
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.kg_operations import save_kg


TYPE_TO_DOMAIN = {
    "Task": {
        "domain_id": "tasks_and_benchmarks",
        "label": "Tasks and Benchmarks",
        "description": "Research tasks, objectives, and benchmark problems.",
    },
    "Method": {
        "domain_id": "methods_and_models",
        "label": "Methods and Models",
        "description": "Algorithms, models, architectures, and techniques.",
    },
    "Metric": {
        "domain_id": "metrics_and_evaluation",
        "label": "Metrics and Evaluation",
        "description": "Evaluation metrics, scores, and performance criteria.",
    },
    "Material": {
        "domain_id": "materials_and_datasets",
        "label": "Materials and Datasets",
        "description": "Datasets, corpora, resources, and experimental materials.",
    },
    "OtherScientificTerm": {
        "domain_id": "scientific_concepts",
        "label": "Scientific Concepts",
        "description": "Scientific concepts, phenomena, and technical terminology.",
    },
}


def canonical_entity_id(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "entity"


def _iter_gold_docs(input_paths: Iterable[str], skip_generic: bool) -> Iterable[Dict[str, Any]]:
    for path in input_paths:
        adapter = SciERCAdapter(path, skip_generic=skip_generic)
        yield from adapter.get_all_gold()


def build_scierc_gold_kg(
    input_paths: Iterable[str],
    skip_generic: bool = True,
) -> Tuple[KnowledgeGraph, OrgChart, Dict[str, Any]]:
    entity_accumulator: Dict[str, Dict[str, Any]] = {}
    triple_accumulator: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for gold_doc in _iter_gold_docs(input_paths, skip_generic=skip_generic):
        doc_key = gold_doc["doc_key"]

        for entity in gold_doc["entities"]:
            label = entity["text"].strip()
            entity_id = canonical_entity_id(label)
            acc = entity_accumulator.setdefault(
                entity_id,
                {
                    "labels": Counter(),
                    "types": Counter(),
                    "doc_keys": set(),
                    "mention_count": 0,
                },
            )
            acc["labels"][label] += 1
            acc["types"][entity["type"]] += 1
            acc["doc_keys"].add(doc_key)
            acc["mention_count"] += 1

        for triple in gold_doc["triples"]:
            subject_id = canonical_entity_id(triple["subject"])
            object_id = canonical_entity_id(triple["object"])
            if subject_id == object_id:
                continue
            key = (subject_id, triple["relation"], object_id)
            acc = triple_accumulator.setdefault(
                key,
                {
                    "doc_keys": set(),
                    "support_count": 0,
                    "subject_type": triple["subject_type"],
                    "object_type": triple["object_type"],
                },
            )
            acc["doc_keys"].add(doc_key)
            acc["support_count"] += 1

    kg = KnowledgeGraph()
    dominant_type_by_entity: Dict[str, str] = {}
    for entity_id, acc in entity_accumulator.items():
        dominant_type = acc["types"].most_common(1)[0][0]
        dominant_type_by_entity[entity_id] = dominant_type
        labels = [label for label, _ in acc["labels"].most_common(5)]
        kg.add_entity(
            entity_id=entity_id,
            labels=labels,
            entity_type=dominant_type,
            metadata={
                "source": "scierc_gold",
                "mention_count": acc["mention_count"],
                "doc_count": len(acc["doc_keys"]),
                "doc_keys": sorted(acc["doc_keys"]),
                "label_counts": dict(acc["labels"]),
                "type_counts": dict(acc["types"]),
            },
        )

    for (subject_id, relation, object_id), acc in triple_accumulator.items():
        kg.add_triple(
            subject=subject_id,
            relation=relation,
            obj=object_id,
            confidence=1.0,
            source="scierc_gold",
            metadata={
                "source": "scierc_gold",
                "support_count": acc["support_count"],
                "doc_count": len(acc["doc_keys"]),
                "doc_keys": sorted(acc["doc_keys"]),
                "subject_type": acc["subject_type"],
                "object_type": acc["object_type"],
            },
        )

    domains: List[Domain] = []
    for entity_type, config in TYPE_TO_DOMAIN.items():
        entity_ids = {
            entity_id
            for entity_id, dominant_type in dominant_type_by_entity.items()
            if dominant_type == entity_type
        }
        if not entity_ids:
            continue

        relation_schema: Dict[str, str] = {}
        for triple in kg.triples:
            if triple.subject in entity_ids or triple.object in entity_ids:
                relation_schema.setdefault(triple.relation, "")

        domains.append(
            Domain(
                domain_id=config["domain_id"],
                label=config["label"],
                description=config["description"],
                entity_ids=entity_ids,
                relation_schema=relation_schema,
                metadata={
                    "source": "scierc_gold",
                    "owner_label": f"{config['label']} Expert",
                    "governance_scope": config["description"],
                    "dominant_entity_type": entity_type,
                },
            )
        )

    entity_to_domain = {}
    for domain in domains:
        for entity_id in domain.entity_ids:
            entity_to_domain[entity_id] = domain.domain_id

    cross_domain = [
        triple
        for triple in kg.triples
        if entity_to_domain.get(triple.subject) != entity_to_domain.get(triple.object)
        and triple.subject in entity_to_domain
        and triple.object in entity_to_domain
    ]
    org_chart = OrgChart(domains=domains, cross_domain_relations=cross_domain)

    stats = {
        "num_entities": len(kg.entities),
        "num_triples": len(kg.triples),
        "num_domains": len(org_chart.domains),
        "domain_sizes": {
            domain.domain_id: len(domain.entity_ids) for domain in org_chart.domains
        },
        "relation_counts": dict(Counter(triple.relation for triple in kg.triples)),
        "entity_type_counts": dict(
            Counter(entity.type for entity in kg.entities.values() if entity.type)
        ),
    }
    return kg, org_chart, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a KG from SciERC gold annotations")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            os.path.join("evaluation", "datasets", "scierc", "train.json"),
            os.path.join("evaluation", "datasets", "scierc", "dev.json"),
            os.path.join("evaluation", "datasets", "scierc", "test.json"),
        ],
        help="SciERC JSONL files to import",
    )
    parser.add_argument(
        "--output-kg",
        default=os.path.join("evaluation", "results", "scierc_gold_kg.json"),
        help="Path to output KG JSON",
    )
    parser.add_argument(
        "--output-org-chart",
        default=os.path.join("evaluation", "results", "scierc_gold_org_chart.json"),
        help="Path to output heuristic org chart JSON",
    )
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Include Generic entities from SciERC",
    )
    args = parser.parse_args()

    kg, org_chart, stats = build_scierc_gold_kg(
        input_paths=args.inputs,
        skip_generic=not args.include_generic,
    )

    kg_dir = os.path.dirname(args.output_kg)
    if kg_dir:
        os.makedirs(kg_dir, exist_ok=True)
    org_dir = os.path.dirname(args.output_org_chart)
    if org_dir:
        os.makedirs(org_dir, exist_ok=True)

    save_kg(kg, args.output_kg)
    with open(args.output_org_chart, "w", encoding="utf-8") as f:
        json.dump(org_chart.to_dict(), f, indent=2)

    print("SciERC gold KG built.")
    print(json.dumps(stats, indent=2))
    print(f"KG written to {args.output_kg}")
    print(f"Org chart written to {args.output_org_chart}")


if __name__ == "__main__":
    main()
