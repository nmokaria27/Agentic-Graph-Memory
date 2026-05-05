#!/usr/bin/env python3
"""Replay governance ablations from a governed KG audit log.

This isolates admission logic from extraction: every condition consumes the
same candidate proposal stream already recorded in the audit log.

Modes:
- domain_rules: keep domain assignment, remove LLM reviewer.
- global_llm_reviewer: remove domain ownership, use one global LLM judge.
- domain_memory_reviewer: keep domain routing and let reviewers use domain-local memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluation"))

from dotenv import load_dotenv

load_dotenv()

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.incremental_enrichment import GOVERNANCE_REVIEW_DECISION_POLICY
from multi_agent_kg.llm.openai_client import chat_completion_json


SCIERC_RELATION_ALIASES = {
    "USED_FOR": "Used-for",
    "USES": "Used-for",
    "UTILIZES": "Used-for",
    "APPLIED_TO": "Used-for",
    "APPLIES_TO": "Used-for",
    "FEATURE_OF": "Feature-of",
    "HAS_FEATURE": "Feature-of",
    "PROPERTY_OF": "Feature-of",
    "PART_OF": "Part-of",
    "IS_PART_OF": "Part-of",
    "COMPONENT_OF": "Part-of",
    "HYPONYM_OF": "Hyponym-of",
    "IS_A": "Hyponym-of",
    "TYPE_OF": "Hyponym-of",
    "SUBTYPE_OF": "Hyponym-of",
    "COMPARE": "Compare",
    "COMPARES": "Compare",
    "COMPARED_TO": "Compare",
    "CONJUNCTION": "Conjunction",
    "AND": "Conjunction",
    "COMBINED_WITH": "Conjunction",
    "EVALUATE_FOR": "Evaluate-for",
    "EVALUATED_FOR": "Evaluate-for",
    "EVALUATES": "Evaluate-for",
}


GLOBAL_REVIEW_PROMPT = """Review candidate SciERC knowledge graph triples.

Use only the candidate triple and its evidence. Do not use domain ownership.

SciERC relation labels:
- Used-for
- Feature-of
- Part-of
- Hyponym-of
- Compare
- Conjunction
- Evaluate-for

Decision policy:
- approve: the evidence supports the triple and the relation is appropriate.
- revise: the evidence supports the fact but subject/object/relation should be normalized.
- reject: the evidence does not support the triple, endpoints are garbage/generic, or relation is only topically plausible.

Return JSON:
{{
  "reviews": [
    {{
      "index": 0,
      "action": "approve|reject|revise",
      "rationale": "brief reason",
      "revised_triple": {{
        "subject": "...",
        "relation": "...",
        "object": "..."
      }}
    }}
  ]
}}

CANDIDATES:
{candidates_json}
"""


MATCHED_GLOBAL_REVIEW_PROMPT = """You are reviewing proposed knowledge-graph updates under the
same admission policy used by the governed domain reviewers.

This ablation intentionally removes governance routing and owner-domain context.
Apply the same review policy, but decide from only the candidate triple,
confidence, source, and evidence.

SciERC relation labels:
- Used-for
- Feature-of
- Part-of
- Hyponym-of
- Compare
- Conjunction
- Evaluate-for

{decision_policy}

Return JSON:
{{
  "reviews": [
    {{
      "index": 0,
      "action": "approve|reject|revise|escalate",
      "rationale": "brief reason",
      "revised_triple": {{
        "subject": "...",
        "relation": "...",
        "object": "..."
      }}
    }}
  ]
}}

CANDIDATES:
{candidates_json}
"""


MINIMAL_GLOBAL_REVIEW_PROMPT = """You are a centralized knowledge-graph triple reviewer.

For each candidate, make only a simple approve/reject decision.
Use only the candidate triple and the evidence text shown with it.
Do not use domain ownership, graph context, neighborhood context, deliberation, or revision.

Approve only if the evidence explicitly supports the subject, relation, and object.
Reject if the evidence is missing, indirect, ambiguous, generic, or only topically related.

Return JSON:
{{
  "reviews": [
    {{
      "index": 0,
      "action": "approve|reject",
      "rationale": "brief reason"
    }}
  ]
}}

CANDIDATES:
{candidates_json}
"""


DOMAIN_MEMORY_REVIEW_PROMPT = """You are a domain expert reviewing proposed updates to your
owned subgraph of a scientific knowledge graph.

Unlike a global reviewer, you may use domain-local memory:
- your domain scope and relation signatures,
- prior accepted/rejected/revised decisions in this domain,
- whether the candidate endpoints already belong to the domain,
- nearby triples already admitted to the domain memory.

Your job is to protect graph quality while preserving useful source-supported facts.

SciERC relation guide:
- Used-for: method/material/term is used for a task, application, or purpose.
- Feature-of: property, feature, representation, or attribute belongs to an entity.
- Part-of: component, module, subproblem, step, or dataset is part of a larger whole.
- Hyponym-of: subject is a kind/type/subclass of object.
- Compare: source compares two methods, systems, models, or results.
- Conjunction: source explicitly groups two peer concepts, methods, tasks, or terms.
- Evaluate-for: method/system/metric is evaluated for a task, dataset, benchmark, or objective.

Decision policy:
- approve: source evidence directly supports this exact subject, relation, and object.
- revise: source evidence supports the fact but the relation or endpoint should be normalized.
- reject: evidence is missing, merely topical, too generic, contradicts domain memory, or uses a relation inconsistent with domain patterns.

Use domain memory to avoid repeating rejected relation mistakes and to normalize facts toward
the domain's accepted relation patterns. Do not use outside background knowledge.

Return JSON:
{{
  "reviews": [
    {{
      "index": 0,
      "action": "approve|reject|revise",
      "rationale": "brief reason that cites evidence and domain memory when relevant",
      "revised_triple": {{
        "subject": "...",
        "relation": "...",
        "object": "..."
      }}
    }}
  ]
}}

DOMAIN MEMORY AND CANDIDATES:
{candidates_json}
"""


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _kg(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("knowledge_graph", data)


def _metadata(triple: Dict[str, Any]) -> Dict[str, Any]:
    value = triple.get("metadata")
    return value if isinstance(value, dict) else {}


def _canonical_relation(relation: Any) -> str:
    raw = str(relation or "").strip()
    norm = raw.upper().replace("-", "_").replace(" ", "_")
    return SCIERC_RELATION_ALIASES.get(norm, raw)


def _copy_entities(kg_data: Dict[str, Any]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for entity in kg_data.get("entities", []):
        kg.add_entity(
            entity_id=entity.get("id"),
            labels=entity.get("labels") or [],
            entity_type=entity.get("type"),
            metadata=entity.get("metadata") or {},
        )
    return kg


def _audit_candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for idx, entry in enumerate(data.get("audit_log", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("triple"), dict):
            continue
        triple = dict(entry["triple"])
        triple["relation"] = _canonical_relation(triple.get("relation"))
        key = (
            triple.get("subject"),
            triple.get("relation"),
            triple.get("object"),
            triple.get("source") or _metadata(triple).get("source_document"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"index": len(candidates), "audit_index": idx, "entry": entry, "triple": triple})
    return candidates


def _add_triple(kg: KnowledgeGraph, triple: Dict[str, Any], *, mode: str, decision: Dict[str, Any]) -> bool:
    subject = triple.get("subject")
    obj = triple.get("object")
    relation = _canonical_relation(triple.get("relation"))
    if not subject or not obj or not relation:
        return False
    metadata = dict(_metadata(triple))
    metadata["ablation_mode"] = mode
    metadata["ablation_action"] = decision.get("action")
    metadata["ablation_rationale"] = decision.get("rationale", "")
    result = kg.add_triple(
        subject=subject,
        relation=relation,
        obj=obj,
        confidence=triple.get("confidence"),
        source=triple.get("source") or metadata.get("source_document"),
        metadata=metadata,
    )
    return result is not None


def _run_domain_rules(
    governed: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    min_confidence: Optional[float],
) -> Tuple[KnowledgeGraph, List[Dict[str, Any]]]:
    kg = _copy_entities(_kg(governed))
    decisions: List[Dict[str, Any]] = []
    for item in candidates:
        entry = item["entry"]
        triple = item["triple"]
        assignment = entry.get("assignment") or {}
        assignment_type = assignment.get("assignment_type", "unowned")
        confidence = triple.get("confidence")
        if min_confidence is not None and (confidence is None or float(confidence) < min_confidence):
            action = "reject"
            rationale = f"Rejected by deterministic confidence floor {min_confidence:.2f}."
        elif assignment_type == "single_owner":
            action = "approve"
            rationale = "Single-owner deterministic rule approved the proposal."
        elif assignment_type == "cross_domain" and len(assignment.get("domain_ids") or []) >= 2:
            action = "approve"
            rationale = "Cross-domain deterministic rule approved because both endpoints have stable owners."
        else:
            action = "reject"
            rationale = "Unowned deterministic rule rejected the proposal."
        decision = {
            "index": item["index"],
            "audit_index": item["audit_index"],
            "action": action,
            "rationale": rationale,
            "assignment_type": assignment_type,
            "triple": triple,
        }
        if action == "approve":
            decision["committed"] = _add_triple(kg, triple, mode="domain_rules", decision=decision)
        else:
            decision["committed"] = False
        decisions.append(decision)
    return kg, decisions


def _domain_lookup(governed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    org = governed.get("org_chart") or {}
    domains = org.get("domains", []) if isinstance(org, dict) else []
    return {
        domain.get("domain_id"): domain
        for domain in domains
        if isinstance(domain, dict) and domain.get("domain_id")
    }


def _primary_domain_id(item: Dict[str, Any]) -> str:
    assignment = item.get("entry", {}).get("assignment") or {}
    return (
        assignment.get("primary_domain_id")
        or (assignment.get("domain_ids") or ["unowned"])[0]
        or "unowned"
    )


def _domain_relation_counts(memory: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for action in ("approve", "reject", "revise"):
        for triple in memory.get(action, []):
            relation = _canonical_relation(triple.get("relation"))
            counts.setdefault(relation, {"approve": 0, "reject": 0, "revise": 0})
            counts[relation][action] += 1
    return counts


def _triple_text(triple: Dict[str, Any]) -> str:
    return f"({triple.get('subject')}) -[{_canonical_relation(triple.get('relation'))}]-> ({triple.get('object')})"


def _neighborhood_lines(kg: KnowledgeGraph, entity_id: str, limit: int = 6) -> List[str]:
    lines: List[str] = []
    triples = kg.triples.values() if hasattr(kg.triples, "values") else kg.triples
    for triple in triples:
        if triple.subject == entity_id or triple.object == entity_id:
            lines.append(f"{triple.subject} -[{triple.relation}]-> {triple.object}")
            if len(lines) >= limit:
                break
    return lines


def _domain_memory_payload(
    item: Dict[str, Any],
    *,
    domain: Dict[str, Any],
    memory: Dict[str, Any],
    kg: KnowledgeGraph,
) -> Dict[str, Any]:
    triple = item["triple"]
    metadata = _metadata(triple)
    domain_entities = set(domain.get("entity_ids") or [])
    relation = _canonical_relation(triple.get("relation"))
    same_relation_approved = [
        _triple_text(t)
        for t in memory.get("approve", [])
        if _canonical_relation(t.get("relation")) == relation
    ][-5:]
    same_relation_rejected = [
        _triple_text(t)
        for t in memory.get("reject", [])
        if _canonical_relation(t.get("relation")) == relation
    ][-5:]
    return {
        "index": item["index"],
        "domain": {
            "domain_id": domain.get("domain_id", "unowned"),
            "label": domain.get("label", "Unowned"),
            "description": domain.get("description", ""),
            "entity_count": len(domain_entities),
            "relation_memory_counts": _domain_relation_counts(memory),
        },
        "assignment": item.get("entry", {}).get("assignment") or {},
        "candidate": {
            "subject": triple.get("subject"),
            "relation": relation,
            "object": triple.get("object"),
            "confidence": triple.get("confidence"),
            "source": triple.get("source") or metadata.get("source_document"),
            "evidence": metadata.get("evidence") or metadata.get("source_sentence") or metadata.get("snippet") or "",
        },
        "endpoint_membership": {
            "subject_in_domain": triple.get("subject") in domain_entities,
            "object_in_domain": triple.get("object") in domain_entities,
        },
        "same_relation_domain_memory": {
            "recent_approved": same_relation_approved,
            "recent_rejected": same_relation_rejected,
        },
        "endpoint_neighborhood": {
            "subject": _neighborhood_lines(kg, str(triple.get("subject") or "")),
            "object": _neighborhood_lines(kg, str(triple.get("object") or "")),
        },
    }


def _review_domain_memory_batch(
    batch: List[Dict[str, Any]],
    *,
    domains: Dict[str, Dict[str, Any]],
    memory_by_domain: Dict[str, Dict[str, Any]],
    kg: KnowledgeGraph,
    model: str,
) -> List[Dict[str, Any]]:
    payload = []
    for item in batch:
        domain_id = _primary_domain_id(item)
        domain = domains.get(domain_id) or {
            "domain_id": domain_id,
            "label": domain_id,
            "description": "No stable owner domain was found.",
            "entity_ids": [],
        }
        memory = memory_by_domain.setdefault(
            domain_id,
            {"approve": [], "reject": [], "revise": []},
        )
        payload.append(_domain_memory_payload(item, domain=domain, memory=memory, kg=kg))

    result = chat_completion_json(
        messages=[
            {
                "role": "system",
                "content": "You are a domain expert reviewing governed KG updates. Return only JSON.",
            },
            {
                "role": "user",
                "content": DOMAIN_MEMORY_REVIEW_PROMPT.format(
                    candidates_json=json.dumps(payload, indent=2)
                ),
            },
        ],
        model=model,
        temperature=0.0,
        max_tokens=4096,
    )
    reviews = result if isinstance(result, list) else result.get("reviews", [])
    by_index = {review.get("index"): review for review in reviews if isinstance(review, dict)}
    output = []
    for item in batch:
        output.append(
            by_index.get(
                item["index"],
                {
                    "index": item["index"],
                    "action": "reject",
                    "rationale": "Domain reviewer omitted this candidate.",
                },
            )
        )
    return output


def _run_domain_memory_reviewer(
    governed: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    model: str,
    batch_size: int,
    min_confidence: Optional[float],
) -> Tuple[KnowledgeGraph, List[Dict[str, Any]]]:
    kg = _copy_entities(_kg(governed))
    domains = _domain_lookup(governed)
    memory_by_domain: Dict[str, Dict[str, Any]] = {}
    decisions: List[Dict[str, Any]] = []
    by_index = {item["index"]: item for item in candidates}
    reviewable = []
    for item in candidates:
        confidence = item["triple"].get("confidence")
        if min_confidence is not None and (confidence is None or float(confidence) < min_confidence):
            decision = {
                "index": item["index"],
                "audit_index": item["audit_index"],
                "action": "reject",
                "rationale": f"Rejected by domain memory confidence floor {min_confidence:.2f}.",
                "triple": item["triple"],
                "committed": False,
                "domain_id": _primary_domain_id(item),
            }
            decisions.append(decision)
            memory_by_domain.setdefault(decision["domain_id"], {"approve": [], "reject": [], "revise": []})["reject"].append(item["triple"])
        else:
            reviewable.append(item)

    for i in range(0, len(reviewable), batch_size):
        batch = reviewable[i : i + batch_size]
        print(f"Domain-memory reviewing batch {i // batch_size + 1}/{(len(reviewable) + batch_size - 1) // batch_size} ({len(batch)} triples)")
        for review in _review_domain_memory_batch(
            batch,
            domains=domains,
            memory_by_domain=memory_by_domain,
            kg=kg,
            model=model,
        ):
            item = by_index[review["index"]]
            action = str(review.get("action", "reject")).lower()
            if action not in {"approve", "reject", "revise"}:
                action = "reject"
            triple = item["triple"]
            if action == "revise" and isinstance(review.get("revised_triple"), dict):
                revised = dict(triple)
                revised.update(
                    {
                        key: value
                        for key, value in review["revised_triple"].items()
                        if key in {"subject", "relation", "object"} and value
                    }
                )
                triple_to_add = revised
            else:
                triple_to_add = triple
            domain_id = _primary_domain_id(item)
            decision = {
                "index": item["index"],
                "audit_index": item["audit_index"],
                "action": action,
                "rationale": review.get("rationale", ""),
                "triple": triple,
                "revised_triple": triple_to_add if action == "revise" else None,
                "domain_id": domain_id,
            }
            if action in {"approve", "revise"}:
                decision["committed"] = _add_triple(
                    kg,
                    triple_to_add,
                    mode="domain_memory_reviewer",
                    decision=decision,
                )
            else:
                decision["committed"] = False
            memory = memory_by_domain.setdefault(domain_id, {"approve": [], "reject": [], "revise": []})
            memory[action].append(triple_to_add if action == "revise" else triple)
            decisions.append(decision)
    decisions.sort(key=lambda item: item["index"])
    return kg, decisions


def _review_batch(
    batch: List[Dict[str, Any]],
    *,
    model: str,
    prompt_mode: str,
) -> List[Dict[str, Any]]:
    payload = []
    for item in batch:
        triple = item["triple"]
        metadata = _metadata(triple)
        candidate = {
            "index": item["index"],
            "subject": triple.get("subject"),
            "relation": _canonical_relation(triple.get("relation")),
            "object": triple.get("object"),
            "source": triple.get("source") or metadata.get("source_document"),
            "evidence": metadata.get("evidence") or metadata.get("source_sentence") or metadata.get("snippet") or "",
        }
        if prompt_mode != "minimal":
            candidate["confidence"] = triple.get("confidence")
        payload.append(candidate)
    if prompt_mode == "matched":
        prompt = MATCHED_GLOBAL_REVIEW_PROMPT.format(
            decision_policy=GOVERNANCE_REVIEW_DECISION_POLICY,
            candidates_json=json.dumps(payload, indent=2),
        )
        system_prompt = (
            "You are a knowledge-graph governance reviewer. "
            "Apply the supplied admission policy and return only JSON."
        )
    elif prompt_mode == "minimal":
        prompt = MINIMAL_GLOBAL_REVIEW_PROMPT.format(candidates_json=json.dumps(payload, indent=2))
        system_prompt = "You are a centralized approve/reject triple reviewer. Return only JSON."
    else:
        prompt = GLOBAL_REVIEW_PROMPT.format(candidates_json=json.dumps(payload, indent=2))
        system_prompt = "You are a strict but fair SciERC KG triple reviewer. Return only JSON."

    result = chat_completion_json(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        temperature=0.0,
        max_tokens=4096,
    )
    reviews = result if isinstance(result, list) else result.get("reviews", [])
    by_index = {review.get("index"): review for review in reviews if isinstance(review, dict)}
    output = []
    for item in batch:
        review = by_index.get(item["index"])
        if review is None:
            review = {
                "index": item["index"],
                "action": "reject",
                "rationale": "Reviewer omitted this candidate.",
            }
        output.append(review)
    return output


def _run_global_llm_reviewer(
    governed: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    model: str,
    batch_size: int,
    min_confidence: Optional[float],
    prompt_mode: str,
) -> Tuple[KnowledgeGraph, List[Dict[str, Any]]]:
    kg = _copy_entities(_kg(governed))
    decisions: List[Dict[str, Any]] = []
    by_index = {item["index"]: item for item in candidates}
    reviewable = []
    for item in candidates:
        confidence = item["triple"].get("confidence")
        if min_confidence is not None and (confidence is None or float(confidence) < min_confidence):
            decisions.append(
                {
                    "index": item["index"],
                    "audit_index": item["audit_index"],
                    "action": "reject",
                    "rationale": f"Rejected by global reviewer confidence floor {min_confidence:.2f}.",
                    "triple": item["triple"],
                    "committed": False,
                }
            )
        else:
            reviewable.append(item)

    for i in range(0, len(reviewable), batch_size):
        batch = reviewable[i : i + batch_size]
        print(f"Reviewing batch {i // batch_size + 1}/{(len(reviewable) + batch_size - 1) // batch_size} ({len(batch)} triples)")
        for review in _review_batch(batch, model=model, prompt_mode=prompt_mode):
            item = by_index[review["index"]]
            action = str(review.get("action", "reject")).lower()
            if action not in {"approve", "reject", "revise"}:
                action = "reject"
            triple = item["triple"]
            if action == "revise" and isinstance(review.get("revised_triple"), dict):
                revised = dict(triple)
                revised.update(
                    {
                        key: value
                        for key, value in review["revised_triple"].items()
                        if key in {"subject", "relation", "object"} and value
                    }
                )
                triple_to_add = revised
            else:
                triple_to_add = triple
            decision = {
                "index": item["index"],
                "audit_index": item["audit_index"],
                "action": action,
                "rationale": review.get("rationale", ""),
                "triple": triple,
                "revised_triple": triple_to_add if action == "revise" else None,
            }
            if action in {"approve", "revise"}:
                decision["committed"] = _add_triple(
                    kg,
                    triple_to_add,
                    mode="global_llm_reviewer",
                    decision=decision,
                )
            else:
                decision["committed"] = False
            decisions.append(decision)
    decisions.sort(key=lambda item: item["index"])
    return kg, decisions


def _decision_counts(decisions: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for decision in decisions:
        action = decision.get("action", "unknown")
        counts[action] = counts.get(action, 0) + 1
    return counts


def _has_evidence(triple: Dict[str, Any]) -> bool:
    metadata = _metadata(triple)
    return bool(
        metadata.get("evidence")
        or metadata.get("source_sentence")
        or metadata.get("snippet")
        or triple.get("evidence")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governed", required=True)
    parser.add_argument("--mode", required=True, choices=["domain_rules", "global_llm_reviewer", "domain_memory_reviewer"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions-output", default="")
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-proposals", type=int, default=0)
    parser.add_argument(
        "--review-prompt-mode",
        choices=["minimal", "strict", "matched"],
        default="strict",
        help=(
            "For global_llm_reviewer: 'minimal' uses a simple approve/reject-only reviewer; "
            "'strict' keeps the older standalone reviewer prompt; "
            "'matched' uses the same admission policy text as the domain governance board, "
            "with routing/domain context removed."
        ),
    )
    args = parser.parse_args()

    governed = _load(args.governed)
    candidates = _audit_candidates(governed)
    if args.max_proposals > 0:
        candidates = candidates[: args.max_proposals]

    if args.mode == "domain_rules":
        kg, decisions = _run_domain_rules(
            governed,
            candidates,
            min_confidence=args.min_confidence,
        )
    elif args.mode == "domain_memory_reviewer":
        kg, decisions = _run_domain_memory_reviewer(
            governed,
            candidates,
            model=args.model,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
        )
    else:
        kg, decisions = _run_global_llm_reviewer(
            governed,
            candidates,
            model=args.model,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            prompt_mode=args.review_prompt_mode,
        )

    output_payload = kg.to_dict()
    output_payload["stats"] = {
        "ablation_mode": args.mode,
        "governed_path": args.governed,
        "candidate_proposals": len(candidates),
        "candidate_proposals_with_evidence": sum(
            1 for item in candidates if _has_evidence(item["triple"])
        ),
        "entities": len(kg.entities),
        "triples": len(kg.triples),
        "decision_counts": _decision_counts(decisions),
        "min_confidence": args.min_confidence,
        "review_prompt_mode": args.review_prompt_mode if args.mode == "global_llm_reviewer" else None,
    }
    Path(args.output).write_text(json.dumps(output_payload, indent=2, default=str))
    if args.decisions_output:
        Path(args.decisions_output).write_text(json.dumps(decisions, indent=2, default=str))
    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(output_payload["stats"], indent=2, default=str))
    print(json.dumps(output_payload["stats"], indent=2))


if __name__ == "__main__":
    main()
