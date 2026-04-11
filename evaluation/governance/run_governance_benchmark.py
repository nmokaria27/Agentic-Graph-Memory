"""
Run a governed-update benchmark over an existing KG and org chart.

This measures two distinct properties:
1. Ownership routing quality: did the system assign an update to the right domain(s)?
2. Governance decisions: did the review board accept true updates and reject corrupted ones?
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multi_agent_kg.core import (
    GovernanceReviewBoard,
    LLMConfig,
    Triple,
    load_governed_kg,
    load_kg,
)
from multi_agent_kg.core.domain_experts import OrgChart


@dataclass(frozen=True)
class GovernanceExample:
    label: str  # positive | negative
    triple: Triple
    expected_domains: List[str]
    expected_assignment_type: str  # single_owner | cross_domain | unowned
    source_triple: Optional[Tuple[str, str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "triple": {
                "subject": self.triple.subject,
                "relation": self.triple.relation,
                "object": self.triple.object,
                "confidence": self.triple.confidence,
            },
            "expected_domains": self.expected_domains,
            "expected_assignment_type": self.expected_assignment_type,
            "source_triple": self.source_triple,
        }


def load_org_chart(path: str, kg) -> OrgChart:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return OrgChart.from_dict(data, kg)


def expected_domains_for_triple(org_chart: OrgChart, triple: Triple) -> List[str]:
    entity_map = org_chart.entity_domain_map()
    return sorted(set(entity_map.get(triple.subject, [])) | set(entity_map.get(triple.object, [])))


def expected_assignment_type(expected_domains: Sequence[str]) -> str:
    if not expected_domains:
        return "unowned"
    if len(expected_domains) == 1:
        return "single_owner"
    return "cross_domain"


def corrupt_triple(triple: Triple, kg, org_chart: OrgChart, rng: random.Random) -> Optional[Triple]:
    entity_ids = list(kg.entities.keys())
    existing = {(t.subject, t.relation, t.object) for t in kg.triples}
    source_domains = set(expected_domains_for_triple(org_chart, triple))

    for _ in range(50):
        replacement = rng.choice(entity_ids)
        replacement_domains = set(org_chart.entity_domain_map().get(replacement, []))
        if replacement == triple.subject or replacement == triple.object:
            continue
        if (source_domains and replacement_domains and source_domains == replacement_domains):
            continue

        candidate = Triple(
            subject=triple.subject,
            relation=triple.relation,
            object=replacement,
            confidence=0.35,
            source="governance_benchmark_corruption",
        )
        if (candidate.subject, candidate.relation, candidate.object) not in existing:
            return candidate

    return None


def build_examples(
    kg,
    org_chart: OrgChart,
    *,
    num_positive: int,
    num_negative: int,
    seed: int,
) -> List[GovernanceExample]:
    rng = random.Random(seed)
    triples = [t for t in kg.triples if expected_domains_for_triple(org_chart, t)]
    rng.shuffle(triples)

    positives: List[GovernanceExample] = []
    negatives: List[GovernanceExample] = []

    for triple in triples:
        if len(positives) < num_positive:
            domains = expected_domains_for_triple(org_chart, triple)
            positives.append(
                GovernanceExample(
                    label="positive",
                    triple=Triple(
                        subject=triple.subject,
                        relation=triple.relation,
                        object=triple.object,
                        confidence=0.95,
                        source="governance_benchmark_positive",
                    ),
                    expected_domains=domains,
                    expected_assignment_type=expected_assignment_type(domains),
                )
            )

        if len(negatives) < num_negative:
            corrupted = corrupt_triple(triple, kg, org_chart, rng)
            if corrupted:
                domains = expected_domains_for_triple(org_chart, corrupted)
                negatives.append(
                    GovernanceExample(
                        label="negative",
                        triple=corrupted,
                        expected_domains=domains,
                        expected_assignment_type=expected_assignment_type(domains),
                        source_triple=(triple.subject, triple.relation, triple.object),
                    )
                )

        if len(positives) >= num_positive and len(negatives) >= num_negative:
            break

    return positives + negatives


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def safe_div(n: float, d: float) -> float:
        return round(n / d, 4) if d else 0.0

    routing_exact = 0
    assignment_exact = 0
    domain_recall_sum = 0.0
    domain_precision_sum = 0.0
    positives = 0
    negatives = 0
    positive_approved = 0
    negative_rejected = 0
    false_accepts = 0
    decisions_present = 0
    cross_domain_total = 0
    cross_domain_routed = 0
    cross_domain_escalated = 0

    for row in rows:
        expected = set(row["expected_domains"])
        predicted = set(row["predicted_domains"])
        overlap = len(expected & predicted)

        routing_exact += int(expected == predicted)
        assignment_exact += int(row["expected_assignment_type"] == row["predicted_assignment_type"])
        domain_recall_sum += (overlap / len(expected)) if expected else 1.0
        domain_precision_sum += (overlap / len(predicted)) if predicted else (1.0 if not expected else 0.0)

        if row["expected_assignment_type"] == "cross_domain":
            cross_domain_total += 1
            cross_domain_routed += int(len(predicted) >= 2 and expected.issubset(predicted))
            cross_domain_escalated += int(row.get("decision") == "escalate")

        if row["label"] == "positive":
            positives += 1
            if row.get("decision") is not None:
                decisions_present += 1
                positive_approved += int(row.get("decision") in {"approve", "keep_new", "keep_both", "merge", "revise"})
        else:
            negatives += 1
            decision = row.get("decision")
            if decision is not None:
                decisions_present += 1
                is_rejected = decision in {"reject", "keep_existing", "escalate"}
                negative_rejected += int(is_rejected)
                false_accepts += int(decision in {"approve", "keep_new", "keep_both", "merge", "revise"})

    metrics = {
        "num_examples": len(rows),
        "routing_exact_match": safe_div(routing_exact, len(rows)),
        "assignment_type_accuracy": safe_div(assignment_exact, len(rows)),
        "expected_domain_recall": safe_div(domain_recall_sum, len(rows)),
        "expected_domain_precision": safe_div(domain_precision_sum, len(rows)),
        "cross_domain_routing_recall": safe_div(cross_domain_routed, cross_domain_total),
        "cross_domain_escalation_accuracy": safe_div(cross_domain_escalated, cross_domain_total),
        "decision_coverage": safe_div(decisions_present, len(rows)),
    }
    if decisions_present:
        metrics.update({
            "accept_recall_positive": safe_div(positive_approved, positives),
            "reject_recall_negative": safe_div(negative_rejected, negatives),
            "false_accept_rate_negative": safe_div(false_accepts, negatives),
        })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run governed-update benchmark")
    parser.add_argument("--kg-path", required=True, help="Path to KG JSON")
    parser.add_argument("--org-chart", help="Path to org chart JSON; if omitted, try the governed KG payload")
    parser.add_argument("--num-positive", type=int, default=25, help="Number of held-out true triples")
    parser.add_argument("--num-negative", type=int, default=25, help="Number of corrupted false triples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--route-only", action="store_true", help="Skip LLM governance review; score routing only")
    parser.add_argument("--output", required=True, help="Where to write benchmark JSON")
    args = parser.parse_args()

    governed_kg = load_governed_kg(args.kg_path)
    kg = governed_kg.kg
    if args.org_chart:
        org_chart = load_org_chart(args.org_chart, kg)
    else:
        org_chart = governed_kg.org_chart
        if not org_chart.domains:
            raise SystemExit("ERROR: no org chart provided and the governed KG has no embedded org chart.")
    examples = build_examples(
        kg,
        org_chart,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed,
    )

    board = None if args.route_only else GovernanceReviewBoard(org_chart, kg, LLMConfig(model=args.model))
    rows: List[Dict[str, Any]] = []

    for example in examples:
        assignment = org_chart.route_triple_for_governance(example.triple)
        decision = None
        rationale = ""
        if board:
            result = board._review_candidate(  # evaluation-only harness
                candidate=example.triple,
                assignment=assignment,
                source_text="",
            )
            decision = result.get("action") or result.get("resolution")
            rationale = result.get("rationale", "")

        rows.append({
            **example.to_dict(),
            "predicted_domains": assignment.domain_ids,
            "predicted_assignment_type": assignment.assignment_type,
            "decision": decision,
            "rationale": rationale,
        })

    payload = {
        "kg_path": args.kg_path,
        "org_chart": args.org_chart,
        "model": args.model,
        "route_only": args.route_only,
        "metrics": compute_metrics(rows),
        "governed_kg_stats": governed_kg.get_stats(),
        "structure_metrics": {
            "domain_coverage": round(
                sum(1 for entity_id in kg.entities if org_chart.entity_domain_map().get(entity_id))
                / max(len(kg.entities), 1),
                4,
            ),
            "governance_completeness": round(
                len(governed_kg.audit_log) / max(len(kg.triples), 1),
                4,
            ),
            "audit_trail_integrity": round(
                len(
                    {
                        (decision.triple.subject, decision.triple.relation, decision.triple.object)
                        for decision in governed_kg.audit_log
                    }
                )
                / max(len(kg.triples), 1),
                4,
            ),
        },
        "examples": rows,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved governance benchmark to {args.output}")


if __name__ == "__main__":
    main()
