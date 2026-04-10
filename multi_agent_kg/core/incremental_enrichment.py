"""
Incremental Knowledge Graph Enrichment Pipeline.

Given an *existing* KG and one or more new documents, this module:
1. Runs extraction on the new documents (reusing the existing pipeline)
2. Computes a diff between the new extractions and the existing KG
3. Routes proposed facts to the owning domain expert(s) for governance review
4. Uses an LLM-backed ConflictResolver agent as a fallback adjudicator
5. Merges the accepted changes into the base KG
6. Returns a structured report of what changed

This is the multi-agent "additive" pathway — the counterpart to the
initial build pipeline in deliberative_orchestrator.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.domain_experts import OrgChart
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.kg_operations import (
    KGDiff,
    compute_diff,
    load_kg,
    merge_kg,
    save_kg,
)
from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator
from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json


class ConflictResolver:
    """
    LLM-backed agent that decides how to handle conflicting triples
    when merging new information into an existing KG.
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config

    def resolve(
        self,
        conflicts: List[tuple],
        source_text: str = "",
    ) -> List[Dict[str, Any]]:
        """
        For each (existing_triple, candidate_triple) pair, decide:
        - keep_existing
        - keep_new
        - keep_both  (both are valid, non-contradictory)
        - merge      (synthesize a better triple)

        Returns a list of resolution dicts.
        """
        if not conflicts:
            return []

        conflict_descriptions = []
        for i, (existing, candidate) in enumerate(conflicts):
            conflict_descriptions.append(
                f"Conflict {i+1}:\n"
                f"  EXISTING: ({existing.subject}) -[{existing.relation}]-> ({existing.object})  "
                f"[confidence={existing.confidence}]\n"
                f"  NEW:      ({candidate.subject}) -[{candidate.relation}]-> ({candidate.object})  "
                f"[confidence={candidate.confidence}]"
            )

        prompt = f"""You are a knowledge graph curator. Given an existing knowledge graph and new
information extracted from a document, some triples conflict (same subject + relation
but different objects).

For each conflict below, decide the best resolution:
- "keep_existing"  — the existing triple is more accurate
- "keep_new"       — the new triple is more accurate or more current
- "keep_both"      — both are valid (e.g., a person can have multiple roles)
- "merge"          — combine into a single improved triple

SOURCE TEXT (for reference):
{source_text[:3000]}

CONFLICTS:
{chr(10).join(conflict_descriptions)}

Respond with a JSON array of objects, one per conflict:
[
  {{
    "conflict_index": 1,
    "resolution": "keep_new",
    "rationale": "...",
    "merged_triple": null
  }}
]

If resolution is "merge", include "merged_triple" with keys: subject, relation, object.
Return ONLY the JSON array."""

        response = chat_completion(
            messages=[
                {"role": "system", "content": "You are a knowledge graph expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_config.model,
            temperature=0.1,
        )

        try:
            resolutions = json.loads(response)
            if not isinstance(resolutions, list):
                resolutions = [resolutions]
        except json.JSONDecodeError:
            # Fallback: keep_higher_confidence
            resolutions = [
                {
                    "conflict_index": i + 1,
                    "resolution": "keep_higher_confidence",
                    "rationale": "LLM response could not be parsed",
                }
                for i in range(len(conflicts))
            ]

        return resolutions


class GovernanceReviewBoard:
    """
    Routes updates to the domain expert(s) that own the affected subgraph.

    This makes governance explicit: even non-conflicting facts must be
    reviewed by the responsible expert before they are merged into memory.
    """

    def __init__(
        self,
        org_chart: OrgChart,
        base_kg: KnowledgeGraph,
        llm_config: LLMConfig,
    ):
        self.org_chart = org_chart
        self.base_kg = base_kg
        self.llm_config = llm_config

    def review_new_triples(
        self,
        triples: List[Any],
        source_text: str = "",
    ) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for triple in triples:
            assignment = self.org_chart.route_triple_for_governance(triple)
            decisions.append(
                self._review_candidate(
                    candidate=triple,
                    assignment=assignment,
                    source_text=source_text,
                )
            )
        return decisions

    def resolve_conflicts(
        self,
        conflicts: List[tuple],
        source_text: str = "",
    ) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for index, (existing, candidate) in enumerate(conflicts, start=1):
            assignment = self.org_chart.route_triple_for_governance(candidate)
            decisions.append(
                self._review_candidate(
                    candidate=candidate,
                    existing=existing,
                    assignment=assignment,
                    source_text=source_text,
                    conflict_index=index,
                )
            )
        return decisions

    def _review_candidate(
        self,
        candidate: Any,
        assignment: Any,
        source_text: str = "",
        existing: Optional[Any] = None,
        conflict_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        owner_domains = [
            self.org_chart.find_domain(domain_id)
            for domain_id in assignment.domain_ids
        ]
        owner_domains = [domain for domain in owner_domains if domain is not None]
        domain_context = "\n\n".join(
            [
                f"OWNER DOMAIN: {domain.owner_label}\n"
                f"SCOPE: {domain.governance_scope}\n"
                f"{domain.subgraph_summary(self.base_kg)}"
                for domain in owner_domains[:2]
            ]
        )
        assignment_dict = assignment.to_dict() if hasattr(assignment, "to_dict") else assignment

        if existing is None:
            prompt = f"""You are reviewing a proposed knowledge-graph update under an
expert-governed memory policy.

GOVERNANCE ROUTING:
{json.dumps(assignment_dict, indent=2)}

{domain_context or "No owning domain was found in the current org chart."}

SOURCE TEXT:
{source_text[:3000]}

PROPOSED TRIPLE:
({candidate.subject}) -[{candidate.relation}]-> ({candidate.object})
[confidence={candidate.confidence}]

Decide one action:
- "approve": accept as-is
- "reject": do not add it
- "revise": accept a better normalized triple
- "escalate": insufficient evidence or ownership ambiguity

Return JSON:
{{
  "action": "approve",
  "rationale": "...",
  "revised_triple": null
}}

If action is "revise", include revised_triple with subject, relation, object.
Return ONLY the JSON."""
        else:
            prompt = f"""You are resolving a conflict in an expert-governed knowledge graph.

GOVERNANCE ROUTING:
{json.dumps(assignment_dict, indent=2)}

{domain_context or "No owning domain was found in the current org chart."}

SOURCE TEXT:
{source_text[:3000]}

EXISTING TRIPLE:
({existing.subject}) -[{existing.relation}]-> ({existing.object})
[confidence={existing.confidence}]

CANDIDATE TRIPLE:
({candidate.subject}) -[{candidate.relation}]-> ({candidate.object})
[confidence={candidate.confidence}]

Decide one resolution:
- "keep_existing"
- "keep_new"
- "keep_both"
- "merge"
- "escalate"

Return JSON:
{{
  "resolution": "keep_existing",
  "rationale": "...",
  "merged_triple": null
}}

If resolution is "merge", include merged_triple with subject, relation, object.
Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a domain governance board for a knowledge graph. "
                            "Be conservative: reject or escalate unsupported updates."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.llm_config.model,
                temperature=0.1,
            )
        except Exception:
            if existing is None:
                action = "approve" if (candidate.confidence or 0.0) >= 0.6 else "reject"
                result = {
                    "action": action,
                    "rationale": "Fallback to confidence threshold because governance review failed.",
                    "revised_triple": None,
                }
            else:
                existing_conf = existing.confidence or 0.0
                candidate_conf = candidate.confidence or 0.0
                resolution = "keep_new" if candidate_conf > existing_conf else "keep_existing"
                result = {
                    "resolution": resolution,
                    "rationale": "Fallback to confidence comparison because governance review failed.",
                    "merged_triple": None,
                }

        result["owner_domains"] = [domain.domain_id for domain in owner_domains]
        result["assignment"] = assignment_dict
        if conflict_index is not None:
            result["conflict_index"] = conflict_index
        return result


class IncrementalEnricher:
    """
    High-level controller for incremental KG enrichment.

    Usage:
        enricher = IncrementalEnricher(base_kg, llm_config)
        report = enricher.add_documents([{"id": "doc2", "text": "..."}])
        enricher.save("updated_kg.json")
    """

    def __init__(
        self,
        base_kg: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
        match_threshold: float = 0.80,
        auto_resolve_conflicts: bool = True,
        org_chart: Optional[OrgChart] = None,
        enable_governance: bool = True,
        governed_kg: Optional[GovernedKnowledgeGraph] = None,
    ):
        if governed_kg is not None:
            base_kg = governed_kg.kg
            org_chart = governed_kg.org_chart
        if base_kg is None:
            raise ValueError("IncrementalEnricher requires either base_kg or governed_kg.")
        self.base_kg = base_kg
        self.governed_kg = governed_kg
        self.llm_config = llm_config or LLMConfig(model="gemma3:27b")
        self.match_threshold = match_threshold
        self.auto_resolve_conflicts = auto_resolve_conflicts
        self.org_chart = org_chart
        self.enable_governance = enable_governance and org_chart is not None
        self.conflict_resolver = ConflictResolver(self.llm_config)
        self.governance_board = (
            GovernanceReviewBoard(org_chart, base_kg, self.llm_config)
            if self.enable_governance and org_chart is not None
            else None
        )
        self.enrichment_log: List[Dict[str, Any]] = []

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        quality_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Process new documents and merge their extractions into the base KG.

        Args:
            documents: List of {"id": str, "text": str, "metadata": dict}
            quality_threshold: Minimum confidence for extraction

        Returns:
            Enrichment report dict.
        """
        start = datetime.now()
        report: Dict[str, Any] = {
            "timestamp": start.isoformat(),
            "documents": len(documents),
            "diffs": [],
            "merge_stats": {},
            "conflicts_resolved": 0,
            "governance": {
                "enabled": self.enable_governance,
                "new_triples_reviewed": 0,
                "new_triples_approved": 0,
                "new_triples_revised": 0,
                "new_triples_rejected": 0,
                "new_triples_escalated": 0,
                "conflicts_reviewed": 0,
                "conflicts_escalated": 0,
                "decisions": [],
            },
        }

        # Step 1: Run the extraction pipeline on new documents into a
        #         *separate* KG so we don't pollute the base yet.
        delta_kg = KnowledgeGraph()
        pipeline = DeliberativeOrchestrator(
            llm_config=self.llm_config,
            knowledge_graph=delta_kg,
            governance_mode="audit_only",
            quality_threshold=quality_threshold,
            max_refinement_iterations=1,
            enable_self_consistency=False,  # Speed: skip broken SC
            enable_open_world=True,
            enable_cross_document=False,
            enable_deliberation=False,  # Speed: skip fake deliberation
        )

        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Extracting from new documents")
        print("=" * 70)

        pipeline.process_corpus(documents)

        # Step 2: Compute diff between delta KG and base KG
        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Computing diff")
        print("=" * 70)

        diff = compute_diff(self.base_kg, delta_kg, self.match_threshold)
        print(diff.summary())
        report["diffs"].append({
            "new_entities": len(diff.new_entities),
            "updated_entities": len(diff.updated_entities),
            "new_triples": len(diff.new_triples),
            "conflicting_triples": len(diff.conflicting_triples),
        })

        source_texts = " ".join(d.get("text", "")[:1000] for d in documents)

        # Step 3: Review new triples with the governing expert(s)
        if diff.new_triples and self.governance_board:
            print(f"\nReviewing {len(diff.new_triples)} proposed triples with domain owners...")
            governance_decisions = self.governance_board.review_new_triples(
                diff.new_triples,
                source_texts,
            )
            approved_triples = []
            from multi_agent_kg.core.knowledge_graph import Triple

            for candidate_t, decision in zip(diff.new_triples, governance_decisions):
                action = decision.get("action", "reject")
                report["governance"]["new_triples_reviewed"] += 1
                report["governance"]["decisions"].append({
                    "triple": {
                        "subject": candidate_t.subject,
                        "relation": candidate_t.relation,
                        "object": candidate_t.object,
                    },
                    "action": action,
                    "owner_domains": decision.get("owner_domains", []),
                    "rationale": decision.get("rationale", ""),
                })

                if action == "approve":
                    approved_triples.append(candidate_t)
                    report["governance"]["new_triples_approved"] += 1
                elif action == "revise" and decision.get("revised_triple"):
                    revised = decision["revised_triple"]
                    approved_triples.append(
                        Triple(
                            subject=revised.get("subject", candidate_t.subject),
                            relation=revised.get("relation", candidate_t.relation),
                            object=revised.get("object", candidate_t.object),
                            confidence=candidate_t.confidence,
                            source=candidate_t.source,
                            metadata=candidate_t.metadata,
                        )
                    )
                    report["governance"]["new_triples_revised"] += 1
                elif action == "escalate":
                    report["governance"]["new_triples_escalated"] += 1
                else:
                    report["governance"]["new_triples_rejected"] += 1

            diff.new_triples = approved_triples

        # Step 4: Resolve conflicts
        conflict_strategy = "keep_higher_confidence"
        if diff.conflicting_triples and self.auto_resolve_conflicts:
            print(f"\nResolving {len(diff.conflicting_triples)} conflicts...")
            if self.governance_board:
                print("Using domain-governance review board...")
                resolutions = self.governance_board.resolve_conflicts(
                    diff.conflicting_triples,
                    source_texts,
                )
                conflict_strategy = "keep_existing"
            else:
                print("Using generic LLM conflict resolver...")
                resolutions = self.conflict_resolver.resolve(
                    diff.conflicting_triples, source_texts
                )
            # Apply resolutions
            resolved_triples = []
            for res in resolutions:
                idx = res.get("conflict_index", 1) - 1
                if idx < len(diff.conflicting_triples):
                    resolution = res.get("resolution", "keep_existing")
                    existing_t, candidate_t = diff.conflicting_triples[idx]
                    if self.governance_board:
                        report["governance"]["conflicts_reviewed"] += 1
                        report["governance"]["decisions"].append({
                            "triple": {
                                "subject": candidate_t.subject,
                                "relation": candidate_t.relation,
                                "object": candidate_t.object,
                            },
                            "action": resolution,
                            "owner_domains": res.get("owner_domains", []),
                            "rationale": res.get("rationale", ""),
                        })

                    if resolution == "keep_new":
                        # Move from conflicting to new
                        diff.new_triples.append(candidate_t)
                        resolved_triples.append(idx)
                    elif resolution == "keep_both":
                        diff.new_triples.append(candidate_t)
                        resolved_triples.append(idx)
                    elif resolution == "merge" and res.get("merged_triple"):
                        from multi_agent_kg.core.knowledge_graph import Triple
                        mt = res["merged_triple"]
                        merged = Triple(
                            subject=mt.get("subject", existing_t.subject),
                            relation=mt.get("relation", existing_t.relation),
                            object=mt.get("object", existing_t.object),
                            confidence=max(
                                existing_t.confidence or 0,
                                candidate_t.confidence or 0,
                            ),
                            source="conflict_resolution",
                        )
                        diff.new_triples.append(merged)
                        resolved_triples.append(idx)
                    elif resolution == "escalate":
                        report["governance"]["conflicts_escalated"] += 1
                    # else: keep_existing, do nothing

            # Remove resolved conflicts
            diff.conflicting_triples = [
                c
                for i, c in enumerate(diff.conflicting_triples)
                if i not in resolved_triples
            ]
            report["conflicts_resolved"] = len(resolved_triples)

        # Step 5: Merge
        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Merging into base KG")
        print("=" * 70)

        merge_stats = merge_kg(self.base_kg, diff, conflict_strategy)
        if self.governed_kg is not None:
            self.governed_kg.org_chart.refresh_cross_domain_relations(self.base_kg)
        report["merge_stats"] = merge_stats
        print(f"  Entities added:   {merge_stats['entities_added']}")
        print(f"  Entities updated: {merge_stats['entities_updated']}")
        print(f"  Triples added:    {merge_stats['triples_added']}")
        print(f"  Conflicts resolved: {merge_stats['conflicts_resolved']}")

        elapsed = (datetime.now() - start).total_seconds()
        report["elapsed_seconds"] = elapsed

        self.enrichment_log.append(report)
        return report

    def save(self, path: str) -> None:
        """Save the enriched KG to disk."""
        save_kg(self.base_kg, path)

    @classmethod
    def from_file(
        cls,
        kg_path: str,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ) -> "IncrementalEnricher":
        """Create an enricher from a previously saved KG file."""
        base_kg = load_kg(kg_path)
        return cls(base_kg=base_kg, llm_config=llm_config, **kwargs)
