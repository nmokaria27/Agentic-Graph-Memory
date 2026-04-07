"""
Domain Expert Architecture for Knowledge-Graph-Grounded QA.

Conceptual hierarchy:

    ┌─────────────────────────────────────────────┐
    │            QA Orchestrator                  │
    │  (routes queries to domain experts)         │
    └────────┬──────────┬──────────┬──────────────┘
             │          │          │
      ┌──────▼───┐ ┌────▼────┐ ┌──▼──────────┐
      │ Domain   │ │ Domain  │ │ Domain      │
      │ Expert A │ │ Expert B│ │ Expert C    │
      │          │ │         │ │             │
      │ sub-     │ │ sub-    │ │ sub-        │
      │ agents   │ │ agents  │ │ agents      │
      └──────────┘ └─────────┘ └─────────────┘

Each Domain Expert:
- Owns a *subgraph* (a slice of the full KG)
- Has a domain-specific relation schema
- Contains topic sub-agents that handle specific queries
- Can quantify *coverage*: how much of a query can it answer?

The QA Orchestrator:
- Maintains an "org chart" of all domain experts
- Decomposes a user query
- Routes sub-questions to relevant experts
- Stitches partial answers into a final response

This module provides:
1. Domain / DomainExpert / SubAgent data structures
2. DomainBuilder — clusters a KG into domains automatically
3. DomainExpertAgent — answers queries from its subgraph
4. QAOrchestrator — the top-level query router
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from multi_agent_kg.core.knowledge_graph import Entity, KnowledgeGraph, Triple
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TopicSubAgent:
    """
    A sub-agent within a domain that handles a specific topic cluster.

    Attributes:
        topic_id:    Unique identifier (e.g., "complement_activation")
        label:       Human-readable name
        description: What this sub-agent covers
        entity_ids:  Set of entity IDs this sub-agent is responsible for
        relation_types: Relation types relevant to this topic
        keywords:    Keyword triggers for routing
    """
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
    """
    A logical grouping of related entities and relations in the KG.

    Attributes:
        domain_id:      Unique identifier (e.g., "cardiovascular_health")
        label:          Human-readable name
        description:    What this domain covers
        entity_ids:     All entity IDs belonging to this domain
        relation_schema: Relation types specific to this domain
        topics:         Topic sub-agents within this domain
        metadata:       Extra info (source documents, confidence, etc.)
    """
    domain_id: str
    label: str
    description: str
    entity_ids: Set[str] = field(default_factory=set)
    relation_schema: Dict[str, str] = field(default_factory=dict)  # relation -> description
    topics: List[TopicSubAgent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def owner_label(self) -> str:
        return self.metadata.get("owner_label") or f"{self.label} Expert"

    @property
    def governance_scope(self) -> str:
        return self.metadata.get("governance_scope") or self.description

    def get_subgraph(self, full_kg: KnowledgeGraph) -> Tuple[List[Entity], List[Triple]]:
        """Extract the subgraph belonging to this domain from the full KG."""
        entities = [
            full_kg.entities[eid]
            for eid in self.entity_ids
            if eid in full_kg.entities
        ]
        triples = [
            t for t in full_kg.triples
            if t.subject in self.entity_ids or t.object in self.entity_ids
        ]
        return entities, triples

    def subgraph_summary(self, full_kg: KnowledgeGraph) -> str:
        """Produce a text summary of this domain's subgraph for LLM context."""
        entities, triples = self.get_subgraph(full_kg)
        lines = [f"Domain: {self.label}", f"Description: {self.description}", ""]
        lines.append(f"Entities ({len(entities)}):")
        for e in entities[:50]:
            type_str = f" [{e.type}]" if e.type else ""
            lines.append(f"  - {e.id}{type_str}")
        if len(entities) > 50:
            lines.append(f"  ... and {len(entities) - 50} more")
        lines.append(f"\nRelationships ({len(triples)}):")
        for t in triples[:80]:
            conf = f" (conf={t.confidence:.2f})" if t.confidence else ""
            lines.append(f"  ({t.subject}) -[{t.relation}]-> ({t.object}){conf}")
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
            topics=[TopicSubAgent.from_dict(t) for t in data.get("topics", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class OrgChart:
    """
    The organization chart mapping domains to their expert agents.
    The QA Orchestrator uses this to decide routing.
    """
    domains: List[Domain] = field(default_factory=list)
    cross_domain_relations: List[Triple] = field(default_factory=list)

    def domain_summary(self) -> str:
        """Text summary for the orchestrator's routing prompt."""
        lines = ["AVAILABLE DOMAIN EXPERTS:", ""]
        for d in self.domains:
            topic_labels = ", ".join(t.label for t in d.topics) if d.topics else "general"
            lines.append(
                f"  [{d.domain_id}] {d.label}: {d.description}"
                f"\n    Topics: {topic_labels}"
                f"\n    Entities: {len(d.entity_ids)}"
                f"\n    Relations: {', '.join(list(d.relation_schema.keys())[:10])}"
            )
            lines.append("")
        if self.cross_domain_relations:
            lines.append(f"Cross-domain relations: {len(self.cross_domain_relations)}")
        return "\n".join(lines)

    def find_domain(self, domain_id: str) -> Optional[Domain]:
        for d in self.domains:
            if d.domain_id == domain_id:
                return d
        return None

    def entity_domain_map(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for domain in self.domains:
            for entity_id in domain.entity_ids:
                mapping.setdefault(entity_id, []).append(domain.domain_id)
        return mapping

    def route_triple_for_governance(self, triple: Triple) -> "GovernanceAssignment":
        """Assign an incoming triple to the domain expert(s) that should review it."""
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

            if score > 0:
                score_breakdown[domain.domain_id] = {
                    "score": score,
                    "reasons": reasons,
                }
                best_score = max(best_score, score)

        if not score_breakdown:
            return GovernanceAssignment(
                assignment_type="unowned",
                primary_domain_id=None,
                domain_ids=[],
                rationale=(
                    f"No domain owns subject '{triple.subject}', object "
                    f"'{triple.object}', or relation '{triple.relation}'."
                ),
                score_breakdown={},
            )

        top_domains = sorted(
            [
                domain_id
                for domain_id, details in score_breakdown.items()
                if details["score"] == best_score
            ]
        )

        if len(top_domains) == 1:
            winner = top_domains[0]
            details = score_breakdown[winner]
            return GovernanceAssignment(
                assignment_type="single_owner",
                primary_domain_id=winner,
                domain_ids=top_domains,
                rationale=(
                    f"{winner} has the strongest ownership signal: "
                    + ", ".join(details["reasons"])
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
        domains = [Domain.from_dict(d) for d in data.get("domains", [])]
        entity_domain_map: Dict[str, str] = {}
        for domain in domains:
            for entity_id in domain.entity_ids:
                entity_domain_map.setdefault(entity_id, domain.domain_id)

        cross_domain = [
            triple
            for triple in kg.triples
            if entity_domain_map.get(triple.subject) != entity_domain_map.get(triple.object)
            and triple.subject in entity_domain_map
            and triple.object in entity_domain_map
        ]
        return cls(domains=domains, cross_domain_relations=cross_domain)


@dataclass
class GovernanceAssignment:
    """
    Ownership routing result for a proposed knowledge-graph update.

    assignment_type:
        - single_owner: one domain expert owns the update
        - cross_domain: several experts must adjudicate it jointly
        - unowned: no current domain clearly owns it
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


# ═══════════════════════════════════════════════════════════════════════════
# Multi-hop graph traversal utilities
# ═══════════════════════════════════════════════════════════════════════════

def find_paths(
    kg: KnowledgeGraph,
    source: str,
    target: str,
    max_hops: int = 3,
) -> List[List[Triple]]:
    """BFS to find all paths from *source* to *target* within *max_hops*.

    Returns a list of paths, where each path is an ordered list of Triple
    objects connecting source → … → target.
    """
    from collections import deque

    # Build adjacency list (both directions for undirected traversal)
    adj: Dict[str, List[Triple]] = {}
    for t in kg.triples:
        adj.setdefault(t.subject, []).append(t)
        adj.setdefault(t.object, []).append(t)

    # Normalise names for fuzzy matching (hyphens, underscores, case)
    from multi_agent_kg.core.kg_operations import normalize_entity_name
    source_n = normalize_entity_name(source)
    target_n = normalize_entity_name(target)

    def _match(eid: str, query: str) -> bool:
        en = normalize_entity_name(eid)
        return en == query or query in en or en in query

    # Identify actual node IDs matching source / target
    source_ids = {eid for eid in adj if _match(eid, source_n)}
    target_ids = {eid for eid in adj if _match(eid, target_n)}

    if not source_ids or not target_ids:
        return []

    paths: List[List[Triple]] = []
    queue: deque = deque()  # (current_node, path_so_far, visited)
    for sid in source_ids:
        queue.append((sid, [], {sid}))

    while queue:
        node, path, visited = queue.popleft()
        if len(path) > max_hops:
            continue
        if node in target_ids and path:
            paths.append(path)
            continue
        for triple in adj.get(node, []):
            # Determine the "other" end of this triple
            if triple.subject == node:
                nxt = triple.object
            else:
                nxt = triple.subject
            if nxt not in visited:
                queue.append((nxt, path + [triple], visited | {nxt}))

    return paths


def paths_to_text(paths: List[List[Triple]]) -> str:
    """Render discovered multi-hop paths as human-readable text."""
    if not paths:
        return "No multi-hop paths found."
    lines = [f"Found {len(paths)} path(s):"]
    for i, path in enumerate(paths[:10], 1):
        hops = " → ".join(
            f"({t.subject}) -[{t.relation}]-> ({t.object})" for t in path
        )
        lines.append(f"  Path {i} ({len(path)} hop{'s' if len(path) > 1 else ''}): {hops}")
    if len(paths) > 10:
        lines.append(f"  ... and {len(paths) - 10} more paths")
    return "\n".join(lines)


def neighbourhood(kg: KnowledgeGraph, entity_id: str, hops: int = 2) -> List[Triple]:
    """Return all triples within *hops* of *entity_id* (BFS expansion).

    Uses aggressive normalization so that entity IDs (``homair``),
    display names (``HOMA-IR``), and snake_case forms (``homa_ir``)
    all match correctly against triple subjects/objects.
    """
    from collections import deque
    from multi_agent_kg.core.kg_operations import normalize_for_matching

    # Pre-build normalized adjacency for fast lookup.
    # Triples may use display names while callers pass entity IDs,
    # so we normalise both sides to a common key.
    adj: Dict[str, List[Tuple[Triple, str]]] = {}  # norm_name → [(triple, other_raw_name)]
    for t in kg.triples:
        sn = normalize_for_matching(t.subject)
        on = normalize_for_matching(t.object)
        adj.setdefault(sn, []).append((t, t.object))
        adj.setdefault(on, []).append((t, t.subject))

    seed = normalize_for_matching(entity_id)
    visited: Set[str] = set()
    frontier: deque = deque([(seed, 0)])
    collected: List[Triple] = []
    seen_triples: Set[int] = set()

    while frontier:
        node_norm, depth = frontier.popleft()
        if node_norm in visited or depth > hops:
            continue
        visited.add(node_norm)
        for triple, other_raw in adj.get(node_norm, []):
            tid = id(triple)
            if tid not in seen_triples:
                seen_triples.add(tid)
                collected.append(triple)
            other_norm = normalize_for_matching(other_raw)
            if other_norm not in visited and depth + 1 <= hops:
                frontier.append((other_norm, depth + 1))
    return collected


# ═══════════════════════════════════════════════════════════════════════════
# Domain Builder — clusters a KG into domains using LLM
# ═══════════════════════════════════════════════════════════════════════════

class DomainBuilder:
    """
    Analyzes a KnowledgeGraph and produces an OrgChart by:
    1. Asking an LLM to identify thematic domains from the entity/relation landscape
    2. Assigning each entity to a domain
    3. Creating TopicSubAgents within each domain
    4. Identifying cross-domain bridge relations
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        target_num_domains: Optional[int] = None,
    ):
        self.llm_config = llm_config
        self.target_num_domains = target_num_domains

    def build(
        self,
        kg: KnowledgeGraph,
        target_num_domains: Optional[int] = None,
    ) -> OrgChart:
        """Build an OrgChart from a KnowledgeGraph."""
        print("\n[DomainBuilder] Analyzing KG structure...")
        requested_domains = target_num_domains or self.target_num_domains

        if requested_domains == 1:
            relation_schema = {triple.relation: "" for triple in kg.triples}
            return OrgChart(
                domains=[
                    Domain(
                        domain_id="global_expert",
                        label="Global Expert",
                        description="Single owner for the entire knowledge graph.",
                        entity_ids=set(kg.entities.keys()),
                        relation_schema=relation_schema,
                        metadata={
                            "owner_label": "Global Expert",
                            "governance_scope": "Owns every entity and relation in the graph.",
                        },
                    )
                ],
                cross_domain_relations=[],
            )

        # Prepare KG summary for the LLM
        kg_summary = self._kg_summary(kg)

        # Step 1: Identify domains
        domains_raw = self._identify_domains(kg_summary, requested_domains)

        # Step 2: Assign entities to domains
        entity_assignments = self._assign_entities(kg, domains_raw)

        # Step 3: Build Domain objects with topics
        org_chart = self._build_org_chart(kg, domains_raw, entity_assignments)

        print(f"[DomainBuilder] Created {len(org_chart.domains)} domains "
              f"with {sum(len(d.topics) for d in org_chart.domains)} topics")

        return org_chart

    def _kg_summary(self, kg: KnowledgeGraph) -> str:
        """Create a compact summary of the KG for prompting."""
        entities_by_type: Dict[str, List[str]] = {}
        for eid, e in kg.entities.items():
            etype = e.type or "untyped"
            entities_by_type.setdefault(etype, []).append(eid)

        relation_types: Dict[str, int] = {}
        for t in kg.triples:
            relation_types[t.relation] = relation_types.get(t.relation, 0) + 1

        lines = [f"Knowledge Graph: {len(kg.entities)} entities, {len(kg.triples)} triples", ""]
        lines.append("Entity types:")
        for etype, eids in entities_by_type.items():
            sample = ", ".join(eids[:8])
            lines.append(f"  {etype} ({len(eids)}): {sample}")

        lines.append("\nRelation types:")
        for rel, count in sorted(relation_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {rel}: {count} triples")

        lines.append("\nSample triples:")
        for t in kg.triples[:30]:
            lines.append(f"  ({t.subject}) -[{t.relation}]-> ({t.object})")

        return "\n".join(lines)

    def _identify_domains(
        self,
        kg_summary: str,
        target_num_domains: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify thematic domains."""
        count_instruction = ""
        if target_num_domains:
            count_instruction = (
                f"\nIdentify EXACTLY {target_num_domains} domains. "
                "Do not return fewer or more unless the KG is truly degenerate."
            )

        prompt = f"""Analyze this knowledge graph and identify the major thematic DOMAINS 
(logical groupings of related entities and relationships).

Each domain should represent a coherent area of knowledge where entities 
are densely connected. Think of domains as "departments" in an organization
that would each need their own expert.
{count_instruction}

KNOWLEDGE GRAPH:
{kg_summary}

For each domain, provide:
- domain_id: snake_case identifier
- label: Human-readable name
- description: What this domain covers (2-3 sentences)
- key_entity_types: Entity types that belong primarily to this domain
- key_relations: Relation types most relevant to this domain
- topics: List of sub-topics within this domain, each with:
  - topic_id, label, description, keywords (search terms)

Return a JSON array:
[
  {{
    "domain_id": "inflammatory_pathways",
    "label": "Inflammatory Pathways",
    "description": "Covers cytokines, complement activation, and immune signaling...",
    "key_entity_types": ["BIOLOGICAL_MARKER", "DISEASE_CONDITION"],
    "key_relations": ["ASSOCIATES_WITH", "MEDIATES"],
    "topics": [
      {{
        "topic_id": "cytokine_signaling",
        "label": "Cytokine Signaling",
        "description": "IL-6, TNF-alpha pathways and their effects",
        "keywords": ["IL-6", "TNF", "cytokine", "inflammation"]
      }}
    ]
  }}
]

Return ONLY the JSON array. Be thorough — every entity should fit into at least one domain."""

        try:
            domains = chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are a knowledge organization expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.2,
            )
            if not isinstance(domains, list):
                domains = [domains]
        except Exception:
            # Fallback: single domain
            domains = [{
                "domain_id": "general",
                "label": "General",
                "description": "All entities and relations",
                "key_entity_types": [],
                "key_relations": [],
                "topics": [],
            }]

        return domains

    def _assign_entities(
        self,
        kg: KnowledgeGraph,
        domains_raw: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Assign each entity to domains using LLM for accurate placement.
        
        Two-pass approach:
        1. Filter garbage entities (empty, numeric-only, generic phrases)
        2. Ask LLM to assign each entity to its PRIMARY domain
        3. Use graph connectivity to add secondary domain membership
           (only when an entity bridges two domains via a triple)
        """
        import re

        assignments: Dict[str, List[str]] = {d["domain_id"]: [] for d in domains_raw}
        if not domains_raw:
            return assignments

        # ── Pass 0: Filter garbage entities ──────────────────────────
        _GARBAGE_PHRASES = {
            "our findings", "this study", "we", "they", "it", "its",
            "the study", "results", "data", "analysis", "the method",
            "the procedure", "the results", "the analysis", "the model",
            "the approach", "the system", "the technique", "the treatment",
            "the patient", "the patients", "the group", "the sample",
            "these findings", "these results", "our study", "our results",
            "the present study", "the current study", "previous studies",
            "placebo", "control group", "baseline", "follow-up",
            "patients", "both groups", "both_groups", "these changes",
            "these_changes", "standard care", "standard_care", "",
        }
        _SKIP_TYPES = {"STATISTICAL_METHOD", "STUDY_DESIGN", "ANALYSIS_TECHNIQUE",
                        "STATISTICAL_MODEL", "ORGANIZATION"}
        valid_entities: List[str] = []
        for eid, entity in kg.entities.items():
            text = (entity.labels[0] if entity.labels else eid).strip()
            if not text:
                continue
            if re.fullmatch(r'\d+', text) and re.fullmatch(r'\d+', eid):
                continue
            if text.lower() in _GARBAGE_PHRASES:
                continue
            if len(text) < 2:
                continue
            if (entity.type or "").upper() in _SKIP_TYPES:
                continue
            valid_entities.append(eid)

        if not valid_entities:
            return assignments

        # ── Pass 1: LLM assigns primary domain (batched) ─────────────
        domain_descriptions = "\n".join(
            f"  {d['domain_id']}: {d.get('label', d['domain_id'])} — {d.get('description', '')}"
            for d in domains_raw
        )
        batch_size = 80  # Process in batches to avoid token limits
        for batch_start in range(0, len(valid_entities), batch_size):
            batch = valid_entities[batch_start:batch_start + batch_size]
            entity_list = "\n".join(
                f"  {eid}: {kg.entities[eid].labels[0] if kg.entities[eid].labels else eid}"
                f" [{kg.entities[eid].type or 'untyped'}]"
                for eid in batch
            )

            prompt = f"""Assign each entity to exactly ONE primary domain.

DOMAINS:
{domain_descriptions}

ENTITIES:
{entity_list}

RULES:
- Every entity MUST be assigned to exactly one domain
- Match entities to domains based on semantic relevance, not just keyword overlap
- Genetic variants (mutations, polymorphisms, alleles) → the genetics/molecular domain if available
- Medications and drugs → therapeutic interventions domain
- Anatomical structures and tissues → anatomical domain
- Distribute entities reasonably — avoid putting everything in one domain

Return JSON:
{{
    "assignments": [
        {{"entity_id": "<eid>", "domain_id": "<domain_id>"}}
    ]
}}

Return ONLY the JSON."""

            try:
                result = chat_completion_json(
                    messages=[
                        {"role": "system", "content": "You are a knowledge organization expert. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.llm_config.model,
                    temperature=0.1,
                )
                for item in result.get("assignments", []):
                    did = item.get("domain_id", "")
                    eid = item.get("entity_id", "")
                    if did in assignments and eid in kg.entities:
                        assignments[did].append(eid)
            except Exception:
                # Fallback: use entity type to assign to best-fit domain
                for eid in batch:
                    etype = (kg.entities[eid].type or "").upper()
                    # Simple type-based heuristic as fallback
                    assigned = False
                    for d in domains_raw:
                        for key_type in d.get("key_entity_types", []):
                            if key_type.upper() == etype:
                                assignments[d["domain_id"]].append(eid)
                                assigned = True
                                break
                        if assigned:
                            break
                    if not assigned:
                        assignments[domains_raw[0]["domain_id"]].append(eid)

        # Catch any unassigned entities — distribute by type similarity, not dump into first
        assigned_eids = {eid for eids in assignments.values() for eid in eids}
        for eid in valid_entities:
            if eid not in assigned_eids:
                # Try to find best domain by entity type match
                etype = (kg.entities[eid].type or "").upper()
                best_domain = domains_raw[0]["domain_id"]
                for d in domains_raw:
                    for key_type in d.get("key_entity_types", []):
                        if key_type.upper() == etype or etype in key_type.upper():
                            best_domain = d["domain_id"]
                            break
                assignments[best_domain].append(eid)

        # ── Pass 2: Add secondary memberships via graph bridges ──────
        # An entity gets a *secondary* membership in another domain only
        # if it participates in 2+ triples with entities primary to that domain.
        # This prevents every cross-domain entity from appearing everywhere.
        primary_domain_of: Dict[str, str] = {}
        for did, eids in assignments.items():
            for eid in eids:
                primary_domain_of[eid] = did

        # Count how many triples each entity has with each foreign domain
        from collections import Counter
        foreign_triple_count: Dict[str, Counter] = {}  # entity -> Counter(domain -> count)
        for t in kg.triples:
            subj_d = primary_domain_of.get(t.subject)
            obj_d = primary_domain_of.get(t.object)
            if subj_d and obj_d and subj_d != obj_d:
                foreign_triple_count.setdefault(t.subject, Counter())[obj_d] += 1
                foreign_triple_count.setdefault(t.object, Counter())[subj_d] += 1

        # Only add secondary membership if entity has 2+ triples with that domain
        for eid, domain_counts in foreign_triple_count.items():
            for did, count in domain_counts.items():
                if count >= 2 and eid not in assignments.get(did, []):
                    assignments.setdefault(did, []).append(eid)

        return assignments

    def _build_org_chart(
        self,
        kg: KnowledgeGraph,
        domains_raw: List[Dict[str, Any]],
        entity_assignments: Dict[str, List[str]],
    ) -> OrgChart:
        """Build the final OrgChart from raw LLM output + entity assignments."""
        domains: List[Domain] = []

        for d_raw in domains_raw:
            did = d_raw["domain_id"]
            entity_ids = set(entity_assignments.get(did, []))

            # Build relation schema from the triples touching this domain
            relation_schema: Dict[str, str] = {}
            for rel in d_raw.get("key_relations", []):
                relation_schema[rel] = ""
            for t in kg.triples:
                if t.subject in entity_ids or t.object in entity_ids:
                    if t.relation not in relation_schema:
                        relation_schema[t.relation] = ""

            # Build topic sub-agents and distribute entities among them
            topics: List[TopicSubAgent] = []
            topic_defs = d_raw.get("topics", [])

            if topic_defs and entity_ids:
                # Use LLM to distribute the domain's entities across topics
                topic_summary = json.dumps(
                    [{"topic_id": t["topic_id"], "label": t["label"],
                      "description": t.get("description", "")}
                     for t in topic_defs],
                    indent=2,
                )
                ent_summary = json.dumps(
                    [{"id": eid,
                      "label": kg.entities[eid].labels[0] if eid in kg.entities and kg.entities[eid].labels else eid}
                     for eid in list(entity_ids)[:80]],
                    indent=2,
                )
                assign_prompt = (
                    f"Assign each entity to the BEST-FIT topic.\n\n"
                    f"TOPICS:\n{topic_summary}\n\n"
                    f"ENTITIES:\n{ent_summary}\n\n"
                    f"Return JSON:\n"
                    f'{{"assignments": [{{"entity_id": "<eid>", "topic_id": "<tid>"}}]}}\n'
                    f"Return ONLY the JSON."
                )
                try:
                    t_result = chat_completion_json(
                        messages=[
                            {"role": "system", "content": "You are a topic classification expert. Return only valid JSON."},
                            {"role": "user", "content": assign_prompt},
                        ],
                        model=self.llm_config.model,
                        temperature=0.1,
                    )
                    topic_entity_map: Dict[str, set] = {t["topic_id"]: set() for t in topic_defs}
                    for a in t_result.get("assignments", []):
                        tid = a.get("topic_id", "")
                        eid = a.get("entity_id", "")
                        if tid in topic_entity_map:
                            topic_entity_map[tid].add(eid)
                except Exception:
                    # Fallback: round-robin
                    topic_entity_map = {t["topic_id"]: set() for t in topic_defs}
                    for i, eid in enumerate(entity_ids):
                        tid = topic_defs[i % len(topic_defs)]["topic_id"]
                        topic_entity_map[tid].add(eid)

                for t_raw in topic_defs:
                    topic = TopicSubAgent(
                        topic_id=t_raw["topic_id"],
                        label=t_raw["label"],
                        description=t_raw.get("description", ""),
                        keywords=t_raw.get("keywords", []),
                        entity_ids=topic_entity_map.get(t_raw["topic_id"], set()),
                    )
                    # Assign relevant relation types
                    for rel in relation_schema:
                        for kw in topic.keywords:
                            if kw.lower() in rel.lower():
                                topic.relation_types.add(rel)
                                break
                    topics.append(topic)
            else:
                # No topics defined — create a catch-all
                for t_raw in topic_defs:
                    topic = TopicSubAgent(
                        topic_id=t_raw["topic_id"],
                        label=t_raw["label"],
                        description=t_raw.get("description", ""),
                        keywords=t_raw.get("keywords", []),
                    )
                    topics.append(topic)

            domain = Domain(
                domain_id=did,
                label=d_raw["label"],
                description=d_raw.get("description", ""),
                entity_ids=entity_ids,
                relation_schema=relation_schema,
                topics=topics,
                metadata={
                    "owner_label": d_raw.get("owner_label", f"{d_raw['label']} Expert"),
                    "governance_scope": d_raw.get("description", ""),
                },
            )
            domains.append(domain)

        # Identify cross-domain relations
        all_domain_entities: Dict[str, str] = {}
        for d in domains:
            for eid in d.entity_ids:
                all_domain_entities[eid] = d.domain_id

        cross_domain = []
        for t in kg.triples:
            subj_domain = all_domain_entities.get(t.subject)
            obj_domain = all_domain_entities.get(t.object)
            if subj_domain and obj_domain and subj_domain != obj_domain:
                cross_domain.append(t)

        return OrgChart(domains=domains, cross_domain_relations=cross_domain)


# ═══════════════════════════════════════════════════════════════════════════
# Domain Expert Agent — answers queries from its subgraph
# ═══════════════════════════════════════════════════════════════════════════

class DomainExpertAgent:
    """
    An agent that has "full knowledge" of a specific domain's subgraph.

    Responsibilities:
    - Receive a query (possibly rewritten by the orchestrator)
    - Determine which topic sub-agents are relevant
    - Retrieve relevant facts from its subgraph
    - Generate an answer grounded in the KG
    - Report a coverage score: how much of the query it could answer
    """

    def __init__(
        self,
        domain: Domain,
        full_kg: KnowledgeGraph,
        llm_config: LLMConfig,
    ):
        self.domain = domain
        self.full_kg = full_kg
        self.llm_config = llm_config

    def answer(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Answer a query using this domain's subgraph + multi-hop reasoning.

        Returns:
            {
                "domain_id": str,
                "answer": str,
                "coverage": float,
                "evidence": [str],
                "topics_used": [str],
                "confidence": float,
                "multi_hop_paths": [str],
            }
        """
        # Get subgraph context
        subgraph_text = self.domain.subgraph_summary(self.full_kg)

        # Identify relevant topics
        relevant_topics = self._route_to_topics(query)
        topic_names = [t.label for t in relevant_topics]

        # ── Multi-hop path discovery ────────────────────────────────
        # Extract entity mentions from the query, then find paths
        # between them in the full KG (not just the subgraph).
        multi_hop_text = ""
        query_entities = self._extract_query_entities(query)
        if len(query_entities) >= 2:
            all_paths = []
            for i in range(len(query_entities)):
                for j in range(i + 1, len(query_entities)):
                    pths = find_paths(
                        self.full_kg, query_entities[i], query_entities[j], max_hops=3
                    )
                    all_paths.extend(pths)
            if all_paths:
                multi_hop_text = (
                    "\n\nMULTI-HOP REASONING PATHS (connections between query entities):\n"
                    + paths_to_text(all_paths)
                )
        # Also expand neighbourhood for single-entity queries
        elif len(query_entities) == 1:
            nbr = neighbourhood(self.full_kg, query_entities[0], hops=2)
            if nbr:
                nbr_lines = [f"  ({t.subject}) -[{t.relation}]-> ({t.object})" for t in nbr[:30]]
                multi_hop_text = (
                    f"\n\nNEIGHBOURHOOD (2-hop) of '{query_entities[0]}':\n"
                    + "\n".join(nbr_lines)
                )

        prompt = f"""You are a domain expert for: {self.domain.label}
Domain description: {self.domain.description}

You have access to the following knowledge from a knowledge graph:

{subgraph_text}
{multi_hop_text}
{f"Additional context: {context}" if context else ""}

QUERY: {query}

Based ONLY on the knowledge graph data above, provide:
1. A concise answer to the query using only claims that are directly supported by
   the evidence above. Prefer 1-3 sentences.
2. If the evidence is incomplete, answer only the supported part and explicitly
   note the gap in a short neutral phrase.
3. Do NOT mention domain IDs, expert agents, routing, or the phrase "knowledge graph".
4. Do NOT speculate, generalize, or add background knowledge.
5. A coverage score (0.0-1.0): what fraction of the query can you fully answer 
   from this domain's knowledge? 1.0 = fully answered, 0.0 = cannot answer at all
6. The specific KG triples (as strings) that support your answer
7. Your confidence in the answer (0.0-1.0)

If the query asks about things outside your domain, say so and set coverage accordingly.

Respond in JSON:
{{
    "answer": "A short evidence-grounded answer.",
    "coverage": 0.65,
    "evidence": ["(entity1) -[relation]-> (entity2)", ...],
    "confidence": 0.7,
    "out_of_scope_aspects": ["list of query aspects not in this domain"]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a domain expert for '{self.domain.label}'. "
                            "Answer queries using ONLY the knowledge graph data provided. "
                            "Use multi-hop reasoning paths when available to explain "
                            "indirect connections. Be precise about what you know and don't."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "Failed to generate answer",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
            }

        result["domain_id"] = self.domain.domain_id
        result["topics_used"] = topic_names
        result["multi_hop_paths"] = multi_hop_text if multi_hop_text else "none"

        # Override LLM-guessed coverage/confidence with computed values
        computed = self._compute_coverage_confidence(query, query_entities)
        if computed["entity_coverage"] > 0 or computed["triple_coverage"] > 0:
            result["coverage"] = computed["coverage"]
            result["confidence"] = computed["confidence"]

        return result

    def _compute_coverage_confidence(
        self, query: str, query_entities: List[str]
    ) -> Dict[str, float]:
        """Compute coverage and confidence from actual KG data.

        coverage = avg(entity_coverage, triple_coverage)
        entity_coverage = fraction of query entities found in domain
        triple_coverage = relevant_triples / max(query_entities, 1)
        confidence = avg triple confidence weighted by coverage
        """
        entities_in_domain = self.domain.entity_ids
        if not query_entities:
            query_entities = self._extract_query_entities(query)

        # Entity coverage: how many query entities exist in this domain
        found = sum(1 for qe in query_entities if qe in entities_in_domain)
        entity_coverage = found / max(len(query_entities), 1)

        # Triple coverage: relevant triples for the query entities
        _, domain_triples = self.domain.get_subgraph(self.full_kg)
        relevant_triples = [
            t for t in domain_triples
            if any(
                qe.lower() in t.subject.lower() or qe.lower() in t.object.lower()
                for qe in query_entities
            )
        ] if query_entities else domain_triples
        triple_coverage = min(1.0, len(relevant_triples) / max(len(query_entities), 1))

        coverage = (entity_coverage + triple_coverage) / 2.0

        # Confidence: average triple confidence weighted by coverage
        if relevant_triples:
            avg_conf = sum(
                t.confidence for t in relevant_triples if t.confidence
            ) / len(relevant_triples)
        else:
            avg_conf = 0.0
        confidence = avg_conf * coverage

        return {
            "entity_coverage": entity_coverage,
            "triple_coverage": triple_coverage,
            "coverage": round(coverage, 3),
            "confidence": round(confidence, 3),
        }

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity IDs from the query by word-boundary matching against KG entities."""
        import re
        query_lower = query.lower()
        matched = []
        for eid, entity in self.full_kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                n_lower = n.lower()
                if len(n_lower) < 3:
                    continue
                # Require word-boundary match to avoid partial substring matches
                # e.g. "insulin" shouldn't match in "insulin resistance" unless
                # "insulin" is the actual entity name
                pattern = r'\b' + re.escape(n_lower) + r'\b'
                if re.search(pattern, query_lower):
                    matched.append(eid)
                    break
        return matched

    def _route_to_topics(self, query: str) -> List[TopicSubAgent]:
        """Find which topic sub-agents are relevant to a query."""
        query_lower = query.lower()
        relevant = []
        for topic in self.domain.topics:
            score = 0
            for kw in topic.keywords:
                if kw.lower() in query_lower:
                    score += 1
            if score > 0:
                relevant.append(topic)

        # If no keyword match, return all topics (let the LLM figure it out)
        return relevant if relevant else self.domain.topics


# ═══════════════════════════════════════════════════════════════════════════
# QA Orchestrator — routes queries across domain experts
# ═══════════════════════════════════════════════════════════════════════════

class QAOrchestrator:
    """
    Top-level query router that:
    1. Decomposes a user query into sub-questions
    2. Routes each sub-question to the relevant domain expert(s)
    3. Collects partial answers
    4. Stitches them into a coherent final response

    This implements the "organization chart" pattern described in the
    project vision.
    """

    def __init__(
        self,
        org_chart: OrgChart,
        full_kg: KnowledgeGraph,
        llm_config: LLMConfig,
    ):
        self.org_chart = org_chart
        self.full_kg = full_kg
        self.llm_config = llm_config

        # Initialize domain expert agents
        self.experts: Dict[str, DomainExpertAgent] = {}
        for domain in org_chart.domains:
            self.experts[domain.domain_id] = DomainExpertAgent(
                domain=domain,
                full_kg=full_kg,
                llm_config=llm_config,
            )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a user question using the domain expert network.

        Returns:
            {
                "question": str,
                "final_answer": str,
                "sub_questions": [...],
                "domain_responses": [...],
                "overall_coverage": float,
                "overall_confidence": float,
            }
        """
        print(f"\n{'='*70}")
        print(f"QA ORCHESTRATOR: Processing query")
        print(f"{'='*70}")
        print(f"Q: {question}\n")

        # Step 1: Decompose query and route to domains
        routing = self._decompose_and_route(question)
        sub_questions = routing.get("sub_questions", [])
        print(f"  Decomposed into {len(sub_questions)} sub-questions")

        # Step 2: Dispatch to domain experts (avoid duplicate domain calls)
        domain_responses: List[Dict[str, Any]] = []
        called_domains: set = set()  # Track (domain_id, sub_question) to avoid duplicates
        for sq in sub_questions:
            sq_text = sq.get("question", question)
            target_domains = sq.get("target_domains", [])
            sq_context = sq.get("context", "")

            print(f"\n  Sub-Q: {sq_text}")
            print(f"  → Routing to: {target_domains}")

            for domain_id in target_domains:
                # Skip if we already called this domain for a very similar sub-question
                call_key = domain_id  # One call per domain per query
                if call_key in called_domains:
                    print(f"    [{domain_id}] skipped (already called)")
                    continue

                expert = self.experts.get(domain_id)
                if expert:
                    response = expert.answer(sq_text, context=sq_context)
                    response["sub_question"] = sq_text
                    domain_responses.append(response)
                    called_domains.add(call_key)
                    print(f"    [{domain_id}] coverage={response.get('coverage', 0):.2f}, "
                          f"confidence={response.get('confidence', 0):.2f}")

        # Step 3: Check cross-domain relations for bridging
        cross_domain_context = self._get_cross_domain_context(question)

        # Step 4: Synthesize final answer
        print(f"\n  Synthesizing final answer from {len(domain_responses)} responses...")
        final = self._synthesize(question, domain_responses, cross_domain_context)

        result = {
            "question": question,
            "final_answer": final.get("answer", ""),
            "sub_questions": sub_questions,
            "domain_responses": domain_responses,
            "overall_coverage": final.get("coverage", 0.0),
            "overall_confidence": final.get("confidence", 0.0),
            "gaps": final.get("gaps", []),
        }

        print(f"\n  Overall coverage: {result['overall_coverage']:.2f}")
        print(f"  Overall confidence: {result['overall_confidence']:.2f}")
        if result["gaps"]:
            print(f"  Knowledge gaps: {result['gaps']}")
        print(f"{'='*70}\n")

        return result

    def _decompose_and_route(self, question: str) -> Dict[str, Any]:
        """Decompose a query into sub-questions and route to domains."""
        org_summary = self.org_chart.domain_summary()

        prompt = f"""You are a query routing agent. Given a user question and a list of 
available domain experts, decompose the question into sub-questions and 
route each to the most relevant domain expert(s).

AVAILABLE DOMAIN EXPERTS:
{org_summary}

USER QUESTION: {question}

Decompose the question into focused sub-questions. For each sub-question,
identify which domain expert(s) should handle it. A sub-question may be
routed to multiple domains if it spans topics.

If the question is simple and maps to a single domain, return just one sub-question.

Respond in JSON:
{{
    "sub_questions": [
        {{
            "question": "What is the role of complement activation in microvascular dysfunction?",
            "target_domains": ["inflammatory_pathways", "cardiovascular_health"],
            "context": "Focus on the mediating pathway between insulin resistance and CFR"
        }}
    ]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query decomposition and routing expert. "
                            "Prefer the fewest domains needed and return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            # Fallback: route narrowly instead of broadcasting to every domain.
            result = {
                "sub_questions": [
                    {
                        "question": question,
                        "target_domains": [d.domain_id for d in self.org_chart.domains[:2]],
                        "context": "",
                    }
                ]
            }

        sub_questions = result.get("sub_questions", [])
        for sq in sub_questions:
            target_domains = []
            for domain_id in sq.get("target_domains", []):
                if domain_id in self.experts and domain_id not in target_domains:
                    target_domains.append(domain_id)
                if len(target_domains) >= 2:
                    break
            if not target_domains and self.org_chart.domains:
                target_domains = [self.org_chart.domains[0].domain_id]
            sq["target_domains"] = target_domains

        return result

    def _get_cross_domain_context(self, question: str) -> str:
        """Get cross-domain relation context + multi-hop paths relevant to the query."""
        lines = []

        # Static cross-domain relations
        if self.org_chart.cross_domain_relations:
            lines.append("Cross-domain relationships:")
            for t in self.org_chart.cross_domain_relations[:30]:
                lines.append(f"  ({t.subject}) -[{t.relation}]-> ({t.object})")

        # Multi-hop: find paths between entities mentioned in the question
        import re as _re
        query_lower = question.lower()
        matched_entities = []
        for eid, entity in self.full_kg.entities.items():
            names = [eid.replace("_", " ")] + entity.labels
            for n in names:
                n_lower = n.lower()
                if len(n_lower) > 2 and _re.search(r'\b' + _re.escape(n_lower) + r'\b', query_lower):
                    matched_entities.append(eid)
                    break

        if len(matched_entities) >= 2:
            for i in range(len(matched_entities)):
                for j in range(i + 1, len(matched_entities)):
                    pths = find_paths(
                        self.full_kg, matched_entities[i], matched_entities[j], max_hops=3
                    )
                    if pths:
                        lines.append(
                            f"\nMulti-hop paths ({matched_entities[i]} → {matched_entities[j]}):"
                        )
                        lines.append(paths_to_text(pths))

        return "\n".join(lines) if lines else ""

    def _synthesize(
        self,
        question: str,
        domain_responses: List[Dict[str, Any]],
        cross_domain_context: str,
    ) -> Dict[str, Any]:
        """Synthesize partial answers from domain experts into a final response."""
        if not domain_responses:
            return {
                "answer": "I don't have enough information in the knowledge graph to answer this question.",
                "coverage": 0.0,
                "confidence": 0.0,
                "gaps": ["No domain experts could provide relevant information"],
            }

        # Format domain responses
        response_texts = []
        for i, resp in enumerate(domain_responses):
            response_texts.append(
                f"Domain Expert [{resp.get('domain_id', '?')}] "
                f"(coverage={resp.get('coverage', 0):.2f}, "
                f"confidence={resp.get('confidence', 0):.2f}):\n"
                f"  Answer: {resp.get('answer', 'N/A')}\n"
                f"  Evidence: {resp.get('evidence', [])}\n"
                f"  Out of scope: {resp.get('out_of_scope_aspects', [])}"
            )

        prompt = f"""You are a knowledge synthesis agent. Multiple domain experts have provided
partial answers to a user's question. Your job is to:

1. Combine their answers into a single, coherent response
2. Resolve any contradictions (prefer higher-confidence answers)
3. Note any gaps — aspects of the question that no expert could answer
4. Compute an overall coverage and confidence score
5. Keep the final answer concise and strictly evidence-grounded

USER QUESTION: {question}

DOMAIN EXPERT RESPONSES:
{chr(10).join(response_texts)}

{f"CROSS-DOMAIN CONTEXT:{chr(10)}{cross_domain_context}" if cross_domain_context else ""}

Synthesize a final answer.

RULES:
- Do NOT mention domain experts, routing, confidence scores, or out-of-scope notes in the answer text.
- Do NOT say "the knowledge graph says" or similar meta-commentary.
- Include only claims that are directly supported by the expert evidence above.
- Prefer the shortest answer that fully covers the supported facts.
- If evidence is weak or missing, state the limitation briefly and stop.

Respond in JSON:
{{
    "answer": "Short final answer here.",
    "coverage": 0.85,
    "confidence": 0.8,
    "gaps": ["aspects of the question that couldn't be answered"]
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledge synthesis expert. Combine partial answers "
                            "into a coherent, well-cited response. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "Failed to synthesize answers",
                "coverage": sum(r.get("coverage", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "confidence": sum(r.get("confidence", 0) for r in domain_responses) / max(len(domain_responses), 1),
                "gaps": [],
            }

        return result
