"""
Domain building for governed knowledge graphs.

This module clusters a KG into governed domains and can also bootstrap an
initial org chart from the schema discovered during extraction, which lets
governance exist during KG creation rather than only after the fact.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.governance import Domain, OrgChart, TopicSubAgent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.llm.openai_client import chat_completion_json


class DomainBuilder:
    """
    Builds domain structure for a KG or bootstraps it from extraction-time schema.
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
                            "seed_entity_types": sorted(
                                {entity.type for entity in kg.entities.values() if entity.type}
                            ),
                            "seed_relation_types": sorted(relation_schema.keys()),
                        },
                    )
                ],
                cross_domain_relations=[],
            )

        kg_summary = self._kg_summary(kg)
        domains_raw = self._identify_domains(kg_summary, requested_domains)
        entity_assignments = self._assign_entities(kg, domains_raw)
        org_chart = self._build_org_chart(kg, domains_raw, entity_assignments)

        print(
            f"[DomainBuilder] Created {len(org_chart.domains)} domains "
            f"with {sum(len(domain.topics) for domain in org_chart.domains)} topics"
        )
        return org_chart

    def bootstrap_from_schema(
        self,
        domain_config: Dict[str, Any],
        target_num_domains: Optional[int] = None,
    ) -> OrgChart:
        """
        Build a lightweight org chart before the KG exists by grouping the
        discovered entity/relation schema into preliminary governed domains.
        """
        entity_types = domain_config.get("entity_types", [])
        relation_types = domain_config.get("relation_types", [])
        requested_domains = target_num_domains or self.target_num_domains

        heuristic_org = self._heuristic_bootstrap_from_schema(domain_config, requested_domains)
        if heuristic_org is not None:
            return heuristic_org

        if requested_domains == 1:
            relation_schema = {
                rel.get("type", rel): ""
                for rel in relation_types
                if rel.get("type", rel)
            }
            return OrgChart(
                domains=[
                    Domain(
                        domain_id="global_expert",
                        label="Global Expert",
                        description=domain_config.get(
                            "domain_description",
                            "Single expert governing the discovered schema.",
                        ),
                        entity_ids=set(),
                        relation_schema=relation_schema,
                        metadata={
                            "owner_label": "Global Expert",
                            "governance_scope": domain_config.get("domain_description", ""),
                            "seed_entity_types": [
                                item.get("type", "")
                                for item in entity_types
                                if item.get("type")
                            ],
                            "seed_relation_types": list(relation_schema.keys()),
                            "bootstrap_source": "schema",
                        },
                    )
                ],
                cross_domain_relations=[],
            )

        prompt = self._schema_bootstrap_prompt(domain_config, requested_domains)
        try:
            domains_raw = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledge governance architect. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
                unwrap_array=True,
            )
            if not isinstance(domains_raw, list):
                domains_raw = [domains_raw]
        except Exception:
            domains_raw = [
                {
                    "domain_id": domain_config.get("primary_domain", "general")
                    .lower()
                    .replace(" ", "_"),
                    "label": domain_config.get("primary_domain", "General"),
                    "description": domain_config.get(
                        "domain_description",
                        "Preliminary governed domain bootstrapped from schema discovery.",
                    ),
                    "key_entity_types": [
                        item.get("type", "")
                        for item in entity_types
                        if item.get("type")
                    ],
                    "key_relations": [
                        item.get("type", "")
                        for item in relation_types
                        if item.get("type")
                    ],
                    "topics": [],
                }
            ]

        domains: List[Domain] = []
        for raw in domains_raw:
            if not isinstance(raw, dict):
                continue
            relation_schema = {
                rel: ""
                for rel in raw.get("key_relations", [])
                if rel
            }
            topics = [
                TopicSubAgent(
                    topic_id=topic["topic_id"],
                    label=topic["label"],
                    description=topic.get("description", ""),
                    keywords=topic.get("keywords", []),
                )
                for topic in raw.get("topics", [])
                if isinstance(topic, dict) and topic.get("topic_id") and topic.get("label")
            ]
            domain_id = (
                raw.get("domain_id")
                or raw.get("id")
                or (raw.get("name") or raw.get("label") or "").lower().replace(" ", "_")
            )
            if not domain_id:
                continue
            label = raw.get("label") or raw.get("name") or domain_id
            domains.append(
                Domain(
                    domain_id=domain_id,
                    label=label,
                    description=raw.get("description", ""),
                    entity_ids=set(),
                    relation_schema=relation_schema,
                    topics=topics,
                    metadata={
                        "owner_label": raw.get("owner_label", f"{label} Expert"),
                        "governance_scope": raw.get("description", ""),
                        "seed_entity_types": raw.get("key_entity_types", []),
                        "seed_relation_types": raw.get("key_relations", []),
                        "bootstrap_source": "schema",
                    },
                )
            )
        return OrgChart(domains=domains, cross_domain_relations=[])

    def _heuristic_bootstrap_from_schema(
        self,
        domain_config: Dict[str, Any],
        target_num_domains: Optional[int],
    ) -> Optional[OrgChart]:
        """
        Fast deterministic bootstrap for compact schemas.

        Preliminary domains exist to support ownership routing during KG creation.
        For small fixed schemas, a deterministic partition is more stable and much
        faster than a heavyweight LLM clustering step.
        """
        if domain_config.get("schema_source") != "fixed_schema_override":
            return None

        entity_types = domain_config.get("entity_types", [])
        relation_types = domain_config.get("relation_types", [])
        if not entity_types:
            return None
        if len(entity_types) > 12 or len(relation_types) > 20:
            return None

        buckets = {
            "methods_and_systems": {
                "label": "Methods and Systems",
                "entity_keywords": {"method", "model", "system", "algorithm", "tool"},
                "relation_keywords": {"used", "feature"},
                "topics": ["models", "algorithms", "systems"],
            },
            "tasks_and_concepts": {
                "label": "Tasks and Concepts",
                "entity_keywords": {"task", "term", "concept", "scientific"},
                "relation_keywords": {"part", "hyponym", "conjunction", "feature"},
                "topics": ["tasks", "concepts", "structures"],
            },
            "resources_and_evaluation": {
                "label": "Resources and Evaluation",
                "entity_keywords": {"material", "dataset", "corpus", "metric", "benchmark", "language"},
                "relation_keywords": {"evaluate", "compare", "used"},
                "topics": ["datasets", "metrics", "benchmarks"],
            },
        }

        entity_assignments: Dict[str, List[str]] = {bucket_id: [] for bucket_id in buckets}
        for entity in entity_types:
            entity_type = entity.get("type", "")
            description = entity.get("description", "")
            text = self._normalize_label(f"{entity_type} {description}")
            scored = []
            for bucket_id, config in buckets.items():
                score = sum(1 for keyword in config["entity_keywords"] if keyword in text)
                if score:
                    scored.append((score, bucket_id))
            if scored:
                scored.sort(reverse=True)
                entity_assignments[scored[0][1]].append(entity_type)
            else:
                entity_assignments["tasks_and_concepts"].append(entity_type)

        relation_assignments: Dict[str, List[str]] = {bucket_id: [] for bucket_id in buckets}
        for relation in relation_types:
            relation_type = relation.get("type", relation)
            description = relation.get("description", "")
            text = self._normalize_label(f"{relation_type} {description}")
            scored = []
            for bucket_id, config in buckets.items():
                score = sum(1 for keyword in config["relation_keywords"] if keyword in text)
                if score:
                    scored.append((score, bucket_id))
            if scored:
                scored.sort(reverse=True)
                relation_assignments[scored[0][1]].append(relation_type)
            else:
                relation_assignments["tasks_and_concepts"].append(relation_type)

        active_domains = [
            bucket_id for bucket_id, assigned in entity_assignments.items() if assigned or relation_assignments[bucket_id]
        ]
        if target_num_domains == 1 or len(active_domains) <= 1:
            return None

        domains: List[Domain] = []
        for bucket_id in active_domains:
            config = buckets[bucket_id]
            domains.append(
                Domain(
                    domain_id=bucket_id,
                    label=config["label"],
                    description=f"Preliminary governed domain for {config['label'].lower()} inferred from schema.",
                    entity_ids=set(),
                    relation_schema={relation: "" for relation in relation_assignments[bucket_id]},
                    topics=[
                        TopicSubAgent(
                            topic_id=f"{bucket_id}_{topic}",
                            label=topic.replace("_", " ").title(),
                            description=f"Bootstrap topic for {topic.replace('_', ' ')}.",
                            keywords=[topic],
                        )
                        for topic in config["topics"]
                    ],
                    metadata={
                        "owner_label": f"{config['label']} Expert",
                        "governance_scope": f"Preliminary bootstrap scope for {config['label'].lower()}",
                        "seed_entity_types": entity_assignments[bucket_id],
                        "seed_relation_types": relation_assignments[bucket_id],
                        "bootstrap_source": "heuristic_schema",
                    },
                )
            )
        return OrgChart(domains=domains, cross_domain_relations=[])

    def assign_entities_to_org_chart(
        self,
        entities: List[Dict[str, Any]],
        org_chart: OrgChart,
        return_diagnostics: bool = False,
    ) -> Any:
        """
        Assign extracted entities to preliminary domains using the seed schema.
        Returns a mapping of entity_id -> candidate domain_ids.
        """
        assignments: Dict[str, List[str]] = {}
        diagnostics = {
            "num_entities": len(entities),
            "assigned_entities": 0,
            "unassigned_entities": 0,
            "multi_assigned_entities": 0,
            "avg_assignment_score": 0.0,
            "assignment_coverage": 0.0,
        }
        if not org_chart.domains:
            return (assignments, diagnostics) if return_diagnostics else assignments

        score_total = 0.0
        for entity in entities:
            entity_id = entity.get("id", entity.get("text", ""))
            entity_text = entity.get("text", entity_id)
            entity_type = entity.get("type") or ""
            entity_tokens = self._tokenize(f"{entity_text} {' '.join(entity.get('mentions', []))}")
            entity_type_norm = self._normalize_label(entity_type)
            scored: List[tuple[float, str]] = []
            for domain in org_chart.domains:
                score = 0.0
                seed_types = [
                    self._normalize_label(item)
                    for item in domain.metadata.get("seed_entity_types", [])
                    if item
                ]
                if entity_type_norm and entity_type_norm in seed_types:
                    score += 5.0
                elif entity_type_norm and any(
                    entity_type_norm in seed_type or seed_type in entity_type_norm
                    for seed_type in seed_types
                ):
                    score += 3.0

                seed_tokens = self._tokenize(
                    " ".join(domain.metadata.get("seed_entity_types", []))
                    + " "
                    + " ".join(domain.metadata.get("seed_relation_types", []))
                    + " "
                    + domain.label
                    + " "
                    + domain.description
                )
                overlap = len(entity_tokens & seed_tokens)
                if overlap:
                    score += min(2.5, overlap * 0.75)

                topic_keywords = {
                    self._normalize_label(keyword)
                    for topic in domain.topics
                    for keyword in topic.keywords
                    if keyword
                }
                if entity_tokens & topic_keywords:
                    score += 1.5

                if score > 0:
                    scored.append((score, domain.domain_id))

            if not scored:
                assignments[entity_id] = []
                diagnostics["unassigned_entities"] += 1
                continue

            scored.sort(reverse=True)
            best_score = scored[0][0]
            threshold = max(best_score - 1.0, best_score * 0.7)
            assignments[entity_id] = [
                domain_id
                for score, domain_id in scored
                if score >= threshold
            ]
            diagnostics["assigned_entities"] += 1
            diagnostics["multi_assigned_entities"] += int(len(assignments[entity_id]) > 1)
            score_total += best_score

        if diagnostics["assigned_entities"]:
            diagnostics["avg_assignment_score"] = round(
                score_total / diagnostics["assigned_entities"],
                4,
            )
        diagnostics["assignment_coverage"] = round(
            diagnostics["assigned_entities"] / max(len(entities), 1),
            4,
        )
        return (assignments, diagnostics) if return_diagnostics else assignments

    def compare_assignment_agreement(
        self,
        reference_org_chart: OrgChart,
        candidate_org_chart: OrgChart,
    ) -> Dict[str, Any]:
        """Compare bootstrap assignments to a later full-build org chart."""
        reference_map = reference_org_chart.entity_domain_map()
        candidate_map = candidate_org_chart.entity_domain_map()
        domain_alignment: Dict[str, str] = {}
        for ref_domain in reference_org_chart.domains:
            best_match = ""
            best_overlap = -1.0
            ref_entities = set(ref_domain.entity_ids)
            for cand_domain in candidate_org_chart.domains:
                cand_entities = set(cand_domain.entity_ids)
                union = ref_entities | cand_entities
                overlap = len(ref_entities & cand_entities) / len(union) if union else 0.0
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = cand_domain.domain_id
            if best_match:
                domain_alignment[ref_domain.domain_id] = best_match

        entity_ids = sorted(set(reference_map) | set(candidate_map))
        if not entity_ids:
            return {"agreement": 0.0, "num_entities": 0}

        matches = 0
        overlap_sum = 0.0
        for entity_id in entity_ids:
            ref = {
                domain_alignment.get(domain_id, domain_id)
                for domain_id in reference_map.get(entity_id, [])
            }
            cand = set(candidate_map.get(entity_id, []))
            if ref == cand:
                matches += 1
            overlap_sum += (len(ref & cand) / len(ref | cand)) if (ref or cand) else 1.0
        return {
            "agreement": round(matches / len(entity_ids), 4),
            "mean_jaccard_overlap": round(overlap_sum / len(entity_ids), 4),
            "num_entities": len(entity_ids),
            "domain_alignment": domain_alignment,
        }

    def _normalize_label(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    def _tokenize(self, text: str) -> set[str]:
        normalized = self._normalize_label(text)
        return {token for token in normalized.split("_") if token}

    def _kg_summary(self, kg: KnowledgeGraph) -> str:
        entities_by_type: Dict[str, List[str]] = {}
        for entity_id, entity in kg.entities.items():
            entity_type = entity.type or "untyped"
            entities_by_type.setdefault(entity_type, []).append(entity_id)

        relation_types: Dict[str, int] = {}
        for triple in kg.triples:
            relation_types[triple.relation] = relation_types.get(triple.relation, 0) + 1

        lines = [f"Knowledge Graph: {len(kg.entities)} entities, {len(kg.triples)} triples", ""]
        lines.append("Entity types:")
        for entity_type, entity_ids in entities_by_type.items():
            sample = ", ".join(entity_ids[:8])
            lines.append(f"  {entity_type} ({len(entity_ids)}): {sample}")

        lines.append("\nRelation types:")
        for relation, count in sorted(relation_types.items(), key=lambda item: -item[1]):
            lines.append(f"  {relation}: {count} triples")

        lines.append("\nSample triples:")
        for triple in kg.triples[:30]:
            lines.append(f"  ({triple.subject}) -[{triple.relation}]-> ({triple.object})")
        return "\n".join(lines)

    def _identify_domains(
        self,
        kg_summary: str,
        target_num_domains: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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

Return a JSON array and return ONLY the JSON array."""

        try:
            domains = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledge organization expert. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.2,
                unwrap_array=True,
            )
            if not isinstance(domains, list):
                domains = [domains]
        except Exception:
            domains = [
                {
                    "domain_id": "general",
                    "label": "General",
                    "description": "All entities and relations",
                    "key_entity_types": [],
                    "key_relations": [],
                    "topics": [],
                }
            ]
        return domains

    def _assign_entities(
        self,
        kg: KnowledgeGraph,
        domains_raw: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        import re
        from collections import Counter

        assignments: Dict[str, List[str]] = {item["domain_id"]: [] for item in domains_raw}
        if not domains_raw:
            return assignments

        garbage_phrases = {
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
        skip_types = {
            "STATISTICAL_METHOD", "STUDY_DESIGN", "ANALYSIS_TECHNIQUE",
            "STATISTICAL_MODEL", "ORGANIZATION",
        }
        valid_entities: List[str] = []
        for entity_id, entity in kg.entities.items():
            text = (entity.labels[0] if entity.labels else entity_id).strip()
            if not text:
                continue
            if re.fullmatch(r"\d+", text) and re.fullmatch(r"\d+", entity_id):
                continue
            if text.lower() in garbage_phrases:
                continue
            if len(text) < 2:
                continue
            if (entity.type or "").upper() in skip_types:
                continue
            valid_entities.append(entity_id)

        if not valid_entities:
            return assignments

        domain_descriptions = "\n".join(
            f"  {item['domain_id']}: {item.get('label', item['domain_id'])} — {item.get('description', '')}"
            for item in domains_raw
        )
        batch_size = 80
        for batch_start in range(0, len(valid_entities), batch_size):
            batch = valid_entities[batch_start:batch_start + batch_size]
            entity_list = "\n".join(
                f"  {entity_id}: {kg.entities[entity_id].labels[0] if kg.entities[entity_id].labels else entity_id}"
                f" [{kg.entities[entity_id].type or 'untyped'}]"
                for entity_id in batch
            )
            prompt = f"""Assign each entity to exactly ONE primary domain.

DOMAINS:
{domain_descriptions}

ENTITIES:
{entity_list}

RULES:
- Every entity MUST be assigned to exactly one domain
- Match entities to domains based on semantic relevance, not just keyword overlap
- Distribute entities reasonably

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
                        {
                            "role": "system",
                            "content": "You are a knowledge organization expert. Return only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=self.llm_config.model,
                    temperature=0.1,
                )
                for item in result.get("assignments", []):
                    domain_id = item.get("domain_id", "")
                    entity_id = item.get("entity_id", "")
                    if domain_id in assignments and entity_id in kg.entities:
                        assignments[domain_id].append(entity_id)
            except Exception:
                for entity_id in batch:
                    entity_type = (kg.entities[entity_id].type or "").upper()
                    assigned = False
                    for raw in domains_raw:
                        for key_type in raw.get("key_entity_types", []):
                            if key_type.upper() == entity_type:
                                assignments[raw["domain_id"]].append(entity_id)
                                assigned = True
                                break
                        if assigned:
                            break
                    if not assigned:
                        assignments[domains_raw[0]["domain_id"]].append(entity_id)

        assigned_ids = {entity_id for entity_ids in assignments.values() for entity_id in entity_ids}
        for entity_id in valid_entities:
            if entity_id in assigned_ids:
                continue
            entity_type = (kg.entities[entity_id].type or "").upper()
            best_domain = domains_raw[0]["domain_id"]
            for raw in domains_raw:
                for key_type in raw.get("key_entity_types", []):
                    if key_type.upper() == entity_type or entity_type in key_type.upper():
                        best_domain = raw["domain_id"]
                        break
            assignments[best_domain].append(entity_id)

        primary_domain_of: Dict[str, str] = {}
        for domain_id, entity_ids in assignments.items():
            for entity_id in entity_ids:
                primary_domain_of[entity_id] = domain_id

        foreign_triple_count: Dict[str, Counter] = {}
        for triple in kg.triples:
            subject_domain = primary_domain_of.get(triple.subject)
            object_domain = primary_domain_of.get(triple.object)
            if subject_domain and object_domain and subject_domain != object_domain:
                foreign_triple_count.setdefault(triple.subject, Counter())[object_domain] += 1
                foreign_triple_count.setdefault(triple.object, Counter())[subject_domain] += 1

        for entity_id, domain_counts in foreign_triple_count.items():
            for domain_id, count in domain_counts.items():
                if count >= 2 and entity_id not in assignments.get(domain_id, []):
                    assignments.setdefault(domain_id, []).append(entity_id)
        return assignments

    def _build_org_chart(
        self,
        kg: KnowledgeGraph,
        domains_raw: List[Dict[str, Any]],
        entity_assignments: Dict[str, List[str]],
    ) -> OrgChart:
        domains: List[Domain] = []
        for raw in domains_raw:
            domain_id = raw["domain_id"]
            entity_ids = set(entity_assignments.get(domain_id, []))
            relation_schema: Dict[str, str] = {}
            for relation in raw.get("key_relations", []):
                relation_schema[relation] = ""
            for triple in kg.triples:
                if triple.subject in entity_ids or triple.object in entity_ids:
                    relation_schema.setdefault(triple.relation, "")

            topics: List[TopicSubAgent] = []
            topic_defs = raw.get("topics", [])
            if topic_defs and entity_ids:
                topic_summary = json.dumps(
                    [
                        {
                            "topic_id": topic["topic_id"],
                            "label": topic["label"],
                            "description": topic.get("description", ""),
                        }
                        for topic in topic_defs
                    ],
                    indent=2,
                )
                entity_summary = json.dumps(
                    [
                        {
                            "id": entity_id,
                            "label": kg.entities[entity_id].labels[0]
                            if entity_id in kg.entities and kg.entities[entity_id].labels
                            else entity_id,
                        }
                        for entity_id in list(entity_ids)[:80]
                    ],
                    indent=2,
                )
                assign_prompt = (
                    "Assign each entity to the BEST-FIT topic.\n\n"
                    f"TOPICS:\n{topic_summary}\n\n"
                    f"ENTITIES:\n{entity_summary}\n\n"
                    'Return JSON: {"assignments": [{"entity_id": "<eid>", "topic_id": "<tid>"}]}\n'
                    "Return ONLY the JSON."
                )
                try:
                    result = chat_completion_json(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a topic classification expert. Return only valid JSON.",
                            },
                            {"role": "user", "content": assign_prompt},
                        ],
                        model=self.llm_config.model,
                        temperature=0.1,
                    )
                    topic_entity_map: Dict[str, set] = {
                        topic["topic_id"]: set() for topic in topic_defs
                    }
                    for assignment in result.get("assignments", []):
                        topic_id = assignment.get("topic_id", "")
                        entity_id = assignment.get("entity_id", "")
                        if topic_id in topic_entity_map:
                            topic_entity_map[topic_id].add(entity_id)
                except Exception:
                    topic_entity_map = {topic["topic_id"]: set() for topic in topic_defs}
                    for index, entity_id in enumerate(entity_ids):
                        topic_id = topic_defs[index % len(topic_defs)]["topic_id"]
                        topic_entity_map[topic_id].add(entity_id)

                for topic_raw in topic_defs:
                    topic = TopicSubAgent(
                        topic_id=topic_raw["topic_id"],
                        label=topic_raw["label"],
                        description=topic_raw.get("description", ""),
                        keywords=topic_raw.get("keywords", []),
                        entity_ids=topic_entity_map.get(topic_raw["topic_id"], set()),
                    )
                    for relation in relation_schema:
                        for keyword in topic.keywords:
                            if keyword.lower() in relation.lower():
                                topic.relation_types.add(relation)
                                break
                    topics.append(topic)
            else:
                for topic_raw in topic_defs:
                    topics.append(
                        TopicSubAgent(
                            topic_id=topic_raw["topic_id"],
                            label=topic_raw["label"],
                            description=topic_raw.get("description", ""),
                            keywords=topic_raw.get("keywords", []),
                        )
                    )

            domains.append(
                Domain(
                    domain_id=domain_id,
                    label=raw["label"],
                    description=raw.get("description", ""),
                    entity_ids=entity_ids,
                    relation_schema=relation_schema,
                    topics=topics,
                    metadata={
                        "owner_label": raw.get("owner_label", f"{raw['label']} Expert"),
                        "governance_scope": raw.get("description", ""),
                        "seed_entity_types": raw.get("key_entity_types", []),
                        "seed_relation_types": raw.get("key_relations", []),
                    },
                )
            )

        org_chart = OrgChart(domains=domains, cross_domain_relations=[])
        org_chart.refresh_cross_domain_relations(kg)
        return org_chart

    def _schema_bootstrap_prompt(
        self,
        domain_config: Dict[str, Any],
        target_num_domains: Optional[int],
    ) -> str:
        count_instruction = ""
        if target_num_domains:
            count_instruction = (
                f"\nCreate EXACTLY {target_num_domains} preliminary domains if the schema supports it."
            )
        entity_types = json.dumps(domain_config.get("entity_types", []), indent=2)
        relation_types = json.dumps(domain_config.get("relation_types", []), indent=2)
        return f"""You are bootstrapping a governed knowledge structure before full extraction.

Build a preliminary domain org chart from this discovered schema.
Each domain should be coherent and own a subset of entity types and relation types.{count_instruction}

PRIMARY DOMAIN: {domain_config.get("primary_domain", "general")}
SUB-DOMAINS: {domain_config.get("sub_domains", [])}
DESCRIPTION: {domain_config.get("domain_description", "")}

ENTITY TYPES:
{entity_types}

RELATION TYPES:
{relation_types}

Return a JSON array:
[
  {{
    "domain_id": "snake_case",
    "label": "Human-readable label",
    "description": "What this governed domain covers",
    "key_entity_types": ["TYPE_A", "TYPE_B"],
    "key_relations": ["REL_A", "REL_B"],
    "topics": [
      {{
        "topic_id": "snake_case",
        "label": "Topic label",
        "description": "Short description",
        "keywords": ["keyword1", "keyword2"]
      }}
    ]
  }}
]

Return ONLY the JSON array."""


__all__ = ["DomainBuilder"]
