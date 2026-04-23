"""
Governance primitives for the governed knowledge graph.

Definition 1 (Governed Knowledge Graph primitives):
  A governed knowledge graph uses a set of domains D to impose ownership over
  entities and triples in a base graph. The org chart stores domain-owned
  subgraphs plus an ownership function phi:E -> 2^D that maps each entity to one
  or more governing domains. Governance assignments operationalize gamma, the
  routing function that maps a candidate triple update to the responsible
  domain(s) before any downstream approval/revision/rejection decision is made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from multi_agent_kg.core.knowledge_graph import Entity, KnowledgeGraph, Triple


def coerce_metadata(value: Any) -> Dict[str, Any]:
    """Return a metadata dict even if upstream passed a JSON string or junk."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass
class TopicSubAgent:
    """A topic-level specialist inside a governed domain."""

    topic_id: str
    label: str
    description: str
    entity_ids: Set[str] = field(default_factory=set)
    relation_types: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "description": self.description,
            "entity_ids": sorted(self.entity_ids),
            "relation_types": sorted(self.relation_types),
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicSubAgent":
        return cls(
            topic_id=data["topic_id"],
            label=data["label"],
            description=data.get("description", ""),
            entity_ids=set(data.get("entity_ids", [])),
            relation_types=set(data.get("relation_types", [])),
            keywords=data.get("keywords", []),
        )


@dataclass
class Domain:
    """A coherent governed area of the graph owned by a domain expert agent."""

    domain_id: str
    label: str
    description: str
    entity_ids: Set[str] = field(default_factory=set)
    relation_schema: Dict[str, str] = field(default_factory=dict)
    topics: List[TopicSubAgent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def owner_label(self) -> str:
        metadata = coerce_metadata(self.metadata)
        return metadata.get("owner_label") or f"{self.label} Expert"

    @property
    def governance_scope(self) -> str:
        metadata = coerce_metadata(self.metadata)
        return metadata.get("governance_scope") or self.description

    def add_entity(self, entity_id: str) -> None:
        self.entity_ids.add(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        self.entity_ids.discard(entity_id)

    def get_subgraph(self, full_kg: KnowledgeGraph) -> Tuple[List[Entity], List[Triple]]:
        entities = [
            full_kg.entities[eid]
            for eid in self.entity_ids
            if eid in full_kg.entities
        ]
        triples = [
            triple
            for triple in full_kg.triples
            if triple.subject in self.entity_ids or triple.object in self.entity_ids
        ]
        return entities, triples

    def subgraph_summary(self, full_kg: KnowledgeGraph) -> str:
        entities, triples = self.get_subgraph(full_kg)
        lines = [f"Domain: {self.label}", f"Description: {self.description}", ""]
        lines.append(f"Entities ({len(entities)}):")
        for entity in entities[:50]:
            type_str = f" [{entity.type}]" if entity.type else ""
            lines.append(f"  - {entity.id}{type_str}")
        if len(entities) > 50:
            lines.append(f"  ... and {len(entities) - 50} more")
        lines.append(f"\nRelationships ({len(triples)}):")
        for triple in triples[:80]:
            conf = f" (conf={triple.confidence:.2f})" if triple.confidence else ""
            lines.append(
                f"  ({triple.subject}) -[{triple.relation}]-> ({triple.object}){conf}"
            )
        if len(triples) > 80:
            lines.append(f"  ... and {len(triples) - 80} more")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "description": self.description,
            "entity_ids": sorted(self.entity_ids),
            "relation_schema": self.relation_schema,
            "topics": [topic.to_dict() for topic in self.topics],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Domain":
        return cls(
            domain_id=data["domain_id"],
            label=data["label"],
            description=data.get("description", ""),
            entity_ids=set(data.get("entity_ids", [])),
            relation_schema=data.get("relation_schema", {}),
            topics=[TopicSubAgent.from_dict(item) for item in data.get("topics", [])],
            metadata=coerce_metadata(data.get("metadata", {})),
        )


@dataclass
class GovernanceAssignment:
    """
    Ownership routing result for a proposed knowledge-graph update.

    assignment_type:
      - single_owner
      - cross_domain
      - unowned
    """

    assignment_type: str
    primary_domain_id: Optional[str]
    domain_ids: List[str] = field(default_factory=list)
    rationale: str = ""
    score_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_type": self.assignment_type,
            "primary_domain_id": self.primary_domain_id,
            "domain_ids": self.domain_ids,
            "rationale": self.rationale,
            "score_breakdown": self.score_breakdown,
        }

    @classmethod
    def unowned(cls, rationale: str = "") -> "GovernanceAssignment":
        return cls(
            assignment_type="unowned",
            primary_domain_id=None,
            domain_ids=[],
            rationale=rationale,
            score_breakdown={},
        )


@dataclass
class OrgChart:
    """The governed domain structure used during ingestion, enrichment, and QA."""

    domains: List[Domain] = field(default_factory=list)
    cross_domain_relations: List[Triple] = field(default_factory=list)
    _entity_domain_map_cache: Optional[Dict[str, List[str]]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def domain_summary(self) -> str:
        lines = ["AVAILABLE DOMAIN EXPERTS:", ""]
        for domain in self.domains:
            topic_labels = ", ".join(topic.label for topic in domain.topics) if domain.topics else "general"
            lines.append(
                f"  [{domain.domain_id}] {domain.label}: {domain.description}"
                f"\n    Topics: {topic_labels}"
                f"\n    Entities: {len(domain.entity_ids)}"
                f"\n    Relations: {', '.join(list(domain.relation_schema.keys())[:10])}"
            )
            lines.append("")
        if self.cross_domain_relations:
            lines.append(f"Cross-domain relations: {len(self.cross_domain_relations)}")
        return "\n".join(lines)

    def find_domain(self, domain_id: str) -> Optional[Domain]:
        for domain in self.domains:
            if domain.domain_id == domain_id:
                return domain
        return None

    def entity_domain_map(self) -> Dict[str, List[str]]:
        if self._entity_domain_map_cache is None:
            mapping: Dict[str, List[str]] = {}
            for domain in self.domains:
                for entity_id in domain.entity_ids:
                    mapping.setdefault(entity_id, []).append(domain.domain_id)
            self._entity_domain_map_cache = mapping
        return self._entity_domain_map_cache

    def assign_entity(self, entity_id: str, domain_ids: List[str]) -> None:
        normalized = []
        for domain_id in domain_ids:
            if domain_id not in normalized:
                normalized.append(domain_id)
        for domain in self.domains:
            if domain.domain_id in normalized:
                domain.add_entity(entity_id)
        self._entity_domain_map_cache = None

    def remove_entity(self, entity_id: str) -> None:
        for domain in self.domains:
            domain.remove_entity(entity_id)
        self._entity_domain_map_cache = None

    def entity_coverage(self) -> Dict[str, Any]:
        mapping = self.entity_domain_map()
        total_assigned = len(mapping)
        multi_domain = sum(1 for domain_ids in mapping.values() if len(domain_ids) > 1)
        return {
            "entities_with_domains": total_assigned,
            "multi_domain_entities": multi_domain,
        }

    def crosses_domains(self, triple: Triple) -> bool:
        entity_map = self.entity_domain_map()
        subject_domains = set(entity_map.get(triple.subject, []))
        object_domains = set(entity_map.get(triple.object, []))
        return bool(subject_domains and object_domains and subject_domains != object_domains)

    def update_cross_domain_relation(self, triple: Triple) -> None:
        if self.crosses_domains(triple):
            if not any(
                existing.subject == triple.subject
                and existing.relation == triple.relation
                and existing.object == triple.object
                for existing in self.cross_domain_relations
            ):
                self.cross_domain_relations.append(triple)

    def refresh_cross_domain_relations(self, kg: KnowledgeGraph) -> None:
        entity_map = self.entity_domain_map()
        self.cross_domain_relations = [
            triple
            for triple in kg.triples
            if entity_map.get(triple.subject)
            and entity_map.get(triple.object)
            and set(entity_map[triple.subject]) != set(entity_map[triple.object])
        ]

    def route_triple_for_governance(self, triple: Triple) -> GovernanceAssignment:
        entity_map = self.entity_domain_map()
        subject_domains = set(entity_map.get(triple.subject, []))
        object_domains = set(entity_map.get(triple.object, []))
        bridged_domains = sorted(subject_domains | object_domains)

        if subject_domains and object_domains and not (subject_domains & object_domains):
            return GovernanceAssignment(
                assignment_type="cross_domain",
                primary_domain_id=bridged_domains[0],
                domain_ids=bridged_domains,
                rationale=(
                    f"Subject '{triple.subject}' and object '{triple.object}' belong to "
                    "different domains, so the update requires joint governance."
                ),
                score_breakdown={
                    domain_id: {
                        "score": 1,
                        "reasons": ["entity participates in cross-domain fact"],
                    }
                    for domain_id in bridged_domains
                },
            )

        score_breakdown: Dict[str, Dict[str, Any]] = {}
        best_score = 0
        for domain in self.domains:
            score = 0
            reasons: List[str] = []
            if triple.subject in domain.entity_ids:
                score += 3
                reasons.append("subject belongs to domain")
            if triple.object in domain.entity_ids:
                score += 2
                reasons.append("object belongs to domain")
            if triple.relation in domain.relation_schema:
                score += 1
                reasons.append("relation is in domain schema")
            domain_metadata = coerce_metadata(domain.metadata)
            seed_relations = set(domain_metadata.get("seed_relation_types", []))
            if triple.relation in seed_relations and "relation is in domain schema" not in reasons:
                score += 1
                reasons.append("relation matches seed schema")
            if score > 0:
                score_breakdown[domain.domain_id] = {"score": score, "reasons": reasons}
                best_score = max(best_score, score)

        if not score_breakdown:
            return GovernanceAssignment.unowned(
                rationale=(
                    f"No domain owns subject '{triple.subject}', object '{triple.object}', "
                    f"or relation '{triple.relation}'."
                )
            )

        top_domains = sorted(
            domain_id
            for domain_id, details in score_breakdown.items()
            if details["score"] == best_score
        )

        if len(top_domains) == 1:
            winner = top_domains[0]
            return GovernanceAssignment(
                assignment_type="single_owner",
                primary_domain_id=winner,
                domain_ids=top_domains,
                rationale=(
                    f"{winner} has the strongest ownership signal: "
                    + ", ".join(score_breakdown[winner]["reasons"])
                ),
                score_breakdown=score_breakdown,
            )

        return GovernanceAssignment(
            assignment_type="cross_domain",
            primary_domain_id=top_domains[0],
            domain_ids=top_domains,
            rationale=(
                "Multiple domains have equally strong ownership signals for "
                f"({triple.subject}) -[{triple.relation}]-> ({triple.object})."
            ),
            score_breakdown=score_breakdown,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": [domain.to_dict() for domain in self.domains],
            "cross_domain_relation_count": len(self.cross_domain_relations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], kg: KnowledgeGraph) -> "OrgChart":
        domains = [Domain.from_dict(item) for item in data.get("domains", [])]
        org_chart = cls(domains=domains, cross_domain_relations=[])
        org_chart.refresh_cross_domain_relations(kg)
        return org_chart


__all__ = [
    "TopicSubAgent",
    "Domain",
    "OrgChart",
    "GovernanceAssignment",
]
