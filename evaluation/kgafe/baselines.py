"""
Evaluation-only QA baselines used in KGAFE ablations.

These wrappers intentionally live outside the main product code because they
exist to answer research questions about *why* a configuration wins or loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
            return {
                "sub_questions": [
                    {
                        "question": benchmark_question.question,
                        "target_domains": expected_domains[:2],
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
