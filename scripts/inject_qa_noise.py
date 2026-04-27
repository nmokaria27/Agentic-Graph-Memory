"""Create a noisy QA-only KG by injecting false in-domain entities/triples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entity_domain_index(org: Dict[str, Any]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for domain in org.get("domains", []):
        for entity_id in domain.get("entity_ids", []):
            index.setdefault(entity_id, domain.get("domain_id", ""))
    return index


def _domain_by_id(org: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {domain.get("domain_id", ""): domain for domain in org.get("domains", [])}


def _add_entity(entities: List[Dict[str, Any]], entity_id: str, label: str, entity_type: str) -> None:
    entities.append(
        {
            "id": entity_id,
            "labels": [label],
            "type": entity_type,
            "metadata": {
                "source_document": "synthetic_noise",
                "confidence": 0.25,
                "synthetic_noise": True,
            },
        }
    )


def _add_to_domain(domain: Dict[str, Any], entity_id: str) -> None:
    ids = domain.setdefault("entity_ids", [])
    if entity_id not in ids:
        ids.append(entity_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", required=True)
    parser.add_argument("--org-chart", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--noise-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-kg", required=True)
    parser.add_argument("--output-org", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    kg_doc = _load(args.kg_path)
    org = _load(args.org_chart)
    questions = _load(args.questions_json)["questions"]
    kg = kg_doc["knowledge_graph"]
    entities: List[Dict[str, Any]] = kg["entities"]
    triples: List[Dict[str, Any]] = kg["triples"]
    entity_ids = {entity["id"] for entity in entities}
    entity_types = {entity["id"]: entity.get("type") or "OtherScientificTerm" for entity in entities}

    entity_to_domain = _entity_domain_index(org)
    domains = _domain_by_id(org)
    domain_ids = [domain_id for domain_id in domains if domain_id]
    if not domain_ids:
        raise SystemExit("No domains found in org chart")

    added_entities = 0
    added_triples = 0
    for question in questions:
        support = question.get("supporting_triples") or []
        involved = question.get("entities_involved") or []
        anchors = list(dict.fromkeys(
            [triple["subject"] for triple in support]
            + [triple["object"] for triple in support]
            + involved
        ))
        anchors = [anchor for anchor in anchors if anchor in entity_ids]
        if not anchors:
            continue

        for idx in range(args.noise_factor):
            anchor = rng.choice(anchors)
            anchor_domain = entity_to_domain.get(anchor) or rng.choice(domain_ids)
            domain = domains[anchor_domain]
            fake_obj = f"noise_{question['question_id']}_{idx}_object"
            fake_subj = f"noise_{question['question_id']}_{idx}_subject"
            relation = rng.choice(support).get("relation") if support else "Used-for"
            anchor_type = entity_types.get(anchor, "OtherScientificTerm")
            _add_entity(entities, fake_obj, f"synthetic distractor object {question['question_id']} {idx}", anchor_type)
            _add_entity(entities, fake_subj, f"synthetic distractor subject {question['question_id']} {idx}", anchor_type)
            _add_to_domain(domain, fake_obj)
            _add_to_domain(domain, fake_subj)
            added_entities += 2

            triples.append(
                {
                    "subject": anchor,
                    "relation": relation,
                    "object": fake_obj,
                    "confidence": 0.25,
                    "source": "synthetic_noise",
                    "metadata": {
                        "synthetic_noise": True,
                        "question_id": question["question_id"],
                        "noise_kind": "anchor_to_fake_object",
                    },
                }
            )
            triples.append(
                {
                    "subject": fake_subj,
                    "relation": relation,
                    "object": anchor,
                    "confidence": 0.25,
                    "source": "synthetic_noise",
                    "metadata": {
                        "synthetic_noise": True,
                        "question_id": question["question_id"],
                        "noise_kind": "fake_subject_to_anchor",
                    },
                }
            )
            added_triples += 2

    kg_doc.setdefault("runtime_config", {})["qa_noise_injection"] = {
        "noise_factor": args.noise_factor,
        "seed": args.seed,
        "added_entities": added_entities,
        "added_triples": added_triples,
        "mode": "hard_in_domain_anchor_noise",
    }
    Path(args.output_kg).write_text(json.dumps(kg_doc, indent=2), encoding="utf-8")
    Path(args.output_org).write_text(json.dumps(org, indent=2), encoding="utf-8")
    print(json.dumps(kg_doc["runtime_config"]["qa_noise_injection"], indent=2))


if __name__ == "__main__":
    main()
