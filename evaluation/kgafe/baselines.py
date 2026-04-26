"""
Evaluation-only QA baselines used in KGAFE ablations.

These wrappers intentionally live outside the main product code because they
exist to answer research questions about *why* a configuration wins or loses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.domain_experts import (
    Domain,
    DomainExpertAgent,
    OrgChart,
    QAOrchestrator,
    find_paths,
    neighbourhood,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.llm.openai_client import chat_completion_json

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None


class OracleDomainWrapper:
    """Force a QA system to use the benchmark's expected domains."""

    def __init__(self, qa_system: Any):
        self.qa_system = qa_system

    def query(self, question: str) -> Dict[str, Any]:
        return self.qa_system.query(question)

    def query_benchmark_question(self, benchmark_question: Any) -> Dict[str, Any]:
        expected_domains = benchmark_question.expected_domains or []
        if not expected_domains:
            return self.qa_system.query(benchmark_question.question)

        if not hasattr(self.qa_system, "_decompose_and_route"):
            return self.qa_system.query(benchmark_question.question)

        original = self.qa_system._decompose_and_route

        def forced_route(*args, **kwargs):
            max_domains = getattr(self.qa_system, "max_routed_domains", 4)
            return {
                "sub_questions": [
                    {
                        "question": benchmark_question.question,
                        "target_domains": expected_domains[:max_domains],
                        "context": "",
                    }
                ]
            }

        self.qa_system._decompose_and_route = forced_route
        try:
            return self.qa_system.query(benchmark_question.question)
        finally:
            self.qa_system._decompose_and_route = original


class PathFocusedExpert(DomainExpertAgent):
    """Single global expert with path/neighbourhood retrieval instead of full-KG dump."""

    def answer(self, query: str, context: str = "") -> Dict[str, Any]:
        query_entities = self._extract_query_entities(query)
        evidence_blocks: List[str] = []

        if len(query_entities) >= 2:
            all_paths = []
            for i in range(len(query_entities)):
                for j in range(i + 1, len(query_entities)):
                    all_paths.extend(
                        find_paths(self.full_kg, query_entities[i], query_entities[j], max_hops=3)
                    )
            if all_paths:
                evidence_blocks.append("MULTI-HOP PATHS:")
                for path in all_paths[:8]:
                    for triple in path:
                        evidence_blocks.append(
                            f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"
                        )
                    evidence_blocks.append("---")

        for entity_id in query_entities[:4]:
            triples = neighbourhood(self.full_kg, entity_id, hops=2)
            if triples:
                evidence_blocks.append(f"NEIGHBOURHOOD OF {entity_id}:")
                for triple in triples[:20]:
                    evidence_blocks.append(
                        f"({triple.subject}) -[{triple.relation}]-> ({triple.object})"
                    )

        if not evidence_blocks:
            evidence_blocks.append("No query-specific graph evidence was found.")

        prompt = f"""You are a graph QA system. Use ONLY the graph evidence below.

GRAPH EVIDENCE:
{chr(10).join(evidence_blocks)}
{f"Additional context: {context}" if context else ""}

QUERY: {query}

Return JSON:
{{
  "answer": "Short evidence-grounded answer.",
  "coverage": 0.0-1.0,
  "evidence": ["(entity) -[relation]-> (entity)"],
  "confidence": 0.0-1.0,
  "out_of_scope_aspects": ["missing aspects"]
}}

Rules:
- Be concise.
- Do not use background knowledge.
- If evidence is weak, answer only the supported part.
- Do not mention expert systems or the knowledge graph.

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer from graph evidence only. "
                            "Be concise, conservative, and return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
                "out_of_scope_aspects": [query],
            }

        result["domain_id"] = self.domain.domain_id
        result["topics_used"] = []
        result["multi_hop_paths"] = "focused"
        return result


class PathFocusedQAOrchestrator(QAOrchestrator):
    """Flat retrieval baseline using a path/neighbourhood-focused global expert."""

    def __init__(
        self,
        org_chart: OrgChart,
        full_kg: KnowledgeGraph,
        llm_config: LLMConfig,
    ):
        super().__init__(org_chart=org_chart, full_kg=full_kg, llm_config=llm_config)
        self.experts = {}
        for domain in org_chart.domains:
            self.experts[domain.domain_id] = PathFocusedExpert(
                domain=domain,
                full_kg=full_kg,
                llm_config=llm_config,
            )


@dataclass
class GraphCommunity:
    community_id: str
    entity_ids: Set[str]
    triples: List[Any]
    summary: str
    keyword_index: Set[str] = field(default_factory=set)


def _normalize_terms(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3
    }


class GraphRAGQAOrchestrator:
    """
    Lightweight GraphRAG-style baseline.

    Builds graph communities, creates deterministic community summaries, retrieves
    the most relevant communities for a query, and answers from those summaries.
    """

    def __init__(self, full_kg: KnowledgeGraph, llm_config: LLMConfig):
        self.full_kg = full_kg
        self.llm_config = llm_config
        self.communities = self._build_communities()

    def _build_communities(self) -> List[GraphCommunity]:
        entity_ids = set(self.full_kg.entities.keys())
        if not entity_ids:
            return []

        if nx is not None:
            graph = nx.Graph()
            graph.add_nodes_from(entity_ids)
            for triple in self.full_kg.triples:
                if triple.subject in entity_ids and triple.object in entity_ids:
                    graph.add_edge(triple.subject, triple.object)
            if graph.number_of_edges() > 0:
                try:
                    raw_communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
                except Exception:
                    raw_communities = [set(component) for component in nx.connected_components(graph)]
            else:
                raw_communities = [{entity_id} for entity_id in entity_ids]
        else:
            adjacency: Dict[str, Set[str]] = defaultdict(set)
            for triple in self.full_kg.triples:
                adjacency[triple.subject].add(triple.object)
                adjacency[triple.object].add(triple.subject)
            seen: Set[str] = set()
            raw_communities: List[Set[str]] = []
            for entity_id in entity_ids:
                if entity_id in seen:
                    continue
                stack = [entity_id]
                component: Set[str] = set()
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    component.add(current)
                    stack.extend(adjacency.get(current, set()) - seen)
                raw_communities.append(component or {entity_id})

        communities: List[GraphCommunity] = []
        for idx, entity_set in enumerate(raw_communities):
            triples = [
                triple
                for triple in self.full_kg.triples
                if triple.subject in entity_set or triple.object in entity_set
            ]
            summary = self._summarize_community(entity_set, triples)
            keyword_index = _normalize_terms(summary)
            for entity_id in entity_set:
                keyword_index.update(_normalize_terms(entity_id.replace("_", " ")))
                entity = self.full_kg.entities.get(entity_id)
                if entity:
                    for label in entity.labels:
                        keyword_index.update(_normalize_terms(label))
            communities.append(
                GraphCommunity(
                    community_id=f"community_{idx}",
                    entity_ids=entity_set,
                    triples=triples,
                    summary=summary,
                    keyword_index=keyword_index,
                )
            )
        return communities

    def _summarize_community(self, entity_ids: Set[str], triples: List[Any]) -> str:
        relation_counts = Counter(triple.relation for triple in triples)
        degree_counts = Counter()
        for triple in triples:
            degree_counts[triple.subject] += 1
            degree_counts[triple.object] += 1

        top_entities = [entity_id for entity_id, _ in degree_counts.most_common(8)] or list(entity_ids)[:8]
        top_relations = [relation for relation, _ in relation_counts.most_common(6)]
        sample_triples = triples[:12]

        lines = [
            f"Community size: {len(entity_ids)} entities, {len(triples)} triples.",
            "Key entities: " + ", ".join(top_entities) if top_entities else "Key entities: none",
            "Common relations: " + ", ".join(top_relations) if top_relations else "Common relations: none",
            "Representative triples:",
        ]
        for triple in sample_triples:
            lines.append(f"- ({triple.subject}) -[{triple.relation}]-> ({triple.object})")
        return "\n".join(lines)

    def _extract_query_entities(self, query: str) -> List[str]:
        query_lower = query.lower()
        matched = []
        for entity_id, entity in self.full_kg.entities.items():
            names = [entity_id.replace("_", " ")] + entity.labels
            for name in names:
                name_lower = name.lower()
                if len(name_lower) < 3:
                    continue
                pattern = r"\b" + re.escape(name_lower) + r"\b"
                if re.search(pattern, query_lower):
                    matched.append(entity_id)
                    break
        return matched

    def _score_community(self, community: GraphCommunity, query: str, query_entities: List[str]) -> Tuple[int, int]:
        entity_overlap = sum(1 for entity_id in query_entities if entity_id in community.entity_ids)
        lexical_overlap = len(_normalize_terms(query) & community.keyword_index)
        return (entity_overlap, lexical_overlap)

    def query(self, question: str) -> Dict[str, Any]:
        query_entities = self._extract_query_entities(question)
        ranked = sorted(
            self.communities,
            key=lambda community: self._score_community(community, question, query_entities),
            reverse=True,
        )
        selected = [community for community in ranked[:3] if community.triples]
        if not selected:
            selected = ranked[:1]

        evidence_lines: List[str] = []
        for community in selected:
            evidence_lines.append(f"[{community.community_id}]")
            evidence_lines.append(community.summary)
            evidence_lines.append("")

        prompt = f"""You are a graph-aware retrieval QA baseline.

You are given retrieved graph-community summaries and representative triples.
Answer the question using ONLY this evidence. Prefer a short direct answer.

RETRIEVED COMMUNITIES:
{chr(10).join(evidence_lines)}

QUESTION: {question}

Return JSON:
{{
  "answer": "Short evidence-grounded answer.",
  "coverage": 0.0-1.0,
  "evidence": ["(entity) -[relation]-> (entity)"],
  "confidence": 0.0-1.0,
  "out_of_scope_aspects": ["missing aspects"]
}}

Rules:
- Do not use outside knowledge.
- If evidence is incomplete, answer only the supported part.
- Be concise.
- Return ONLY valid JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict graph QA system. "
                            "Answer only from the retrieved community evidence and return JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            result = {
                "answer": "",
                "coverage": 0.0,
                "evidence": [],
                "confidence": 0.0,
                "out_of_scope_aspects": [question],
            }

        result["final_answer"] = result.get("answer", "")
        result["community_ids"] = [community.community_id for community in selected]
        result["retrieval_mode"] = "community_summary"
        return result


@dataclass(frozen=True)
class BaselineBuildResult:
    qa_system: Any
    org_chart: OrgChart


def build_baseline_system(
    baseline_name: str,
    *,
    kg: KnowledgeGraph,
    llm_config: LLMConfig,
    org_chart: OrgChart,
    advanced_orchestrator_cls: Any,
) -> BaselineBuildResult:
    """Construct special-case ablation baselines outside the main QA stack."""
    if baseline_name == "oracle_domain_basic":
        system = QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=llm_config)
        return BaselineBuildResult(qa_system=OracleDomainWrapper(system), org_chart=org_chart)

    if baseline_name == "flat_path_basic":
        global_domain = Domain(
            domain_id="global_path_expert",
            label="Global Path Expert",
            description="Single global expert with path-focused retrieval.",
            entity_ids=set(kg.entities.keys()),
            relation_schema={triple.relation: triple.relation for triple in kg.triples},
            topics=[],
        )
        global_chart = OrgChart(domains=[global_domain], cross_domain_relations=[])
        system = PathFocusedQAOrchestrator(
            org_chart=global_chart,
            full_kg=kg,
            llm_config=llm_config,
        )
        return BaselineBuildResult(qa_system=system, org_chart=global_chart)

    if baseline_name == "graphrag_basic":
        global_domain = Domain(
            domain_id="global_graphrag",
            label="GraphRAG Baseline",
            description="Graph-aware retrieval baseline using community summaries.",
            entity_ids=set(kg.entities.keys()),
            relation_schema={triple.relation: triple.relation for triple in kg.triples},
            topics=[],
        )
        global_chart = OrgChart(domains=[global_domain], cross_domain_relations=[])
        system = GraphRAGQAOrchestrator(
            full_kg=kg,
            llm_config=llm_config,
        )
        return BaselineBuildResult(qa_system=system, org_chart=global_chart)

    if baseline_name == "oracle_domain_advanced":
        system = advanced_orchestrator_cls(
            org_chart=org_chart,
            full_kg=kg,
            llm_config=llm_config,
            max_exploration_rounds=3,
            enable_debate=True,
            enable_critic=True,
        )
        return BaselineBuildResult(qa_system=OracleDomainWrapper(system), org_chart=org_chart)

    raise ValueError(f"Unknown baseline: {baseline_name}")
