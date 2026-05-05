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
from multi_agent_kg.core.domain_builder import DomainBuilder
from multi_agent_kg.core.domain_experts import OrgChart
from multi_agent_kg.core.governance import coerce_metadata
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph, GovernanceDecision
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.kg_operations import (
    KGDiff,
    compute_diff,
    load_kg,
    merge_kg,
    save_kg,
    save_governed_kg,
)
from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator
from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json


GOVERNANCE_REVIEW_DECISION_POLICY = """Decision policy:
- "approve": accept as-is only if the source text directly supports the subject, relation, and object.
- "reject": do not add it if evidence is missing, merely topical, too generic, or does not support the exact relation.
- "revise": accept a corrected/normalized triple only when the correction is directly supported by the source text.
- "escalate": use only for genuine ownership ambiguity or evidence ambiguity that cannot be safely resolved.

Review rules:
- Do not approve a triple because it is plausible from background knowledge.
- Prefer reject over approve when source support is weak.
- Prefer revise over reject when the evidence clearly supports the same fact with a normalized endpoint or relation.
- Keep revised triples within the same source-supported claim; do not invent a new fact."""


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
        neighborhood_context = self._entity_neighborhood_context(candidate)

        if existing is None:
            prompt = f"""You are reviewing a proposed knowledge-graph update under an
expert-governed memory policy.

GOVERNANCE ROUTING:
{json.dumps(assignment_dict, indent=2)}

{domain_context or "No owning domain was found in the current org chart."}

NEIGHBORHOOD CONTEXT:
{neighborhood_context or "No prior graph context exists for this candidate's endpoints."}

SOURCE TEXT:
{source_text[:3000]}

PROPOSED TRIPLE:
({candidate.subject}) -[{candidate.relation}]-> ({candidate.object})
[confidence={candidate.confidence}]

{GOVERNANCE_REVIEW_DECISION_POLICY}

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

NEIGHBORHOOD CONTEXT:
{neighborhood_context or "No prior graph context exists for this candidate's endpoints."}

SOURCE TEXT:
{source_text[:3000]}

EXISTING TRIPLE:
({existing.subject}) -[{existing.relation}]-> ({existing.object})
[confidence={existing.confidence}]

CANDIDATE TRIPLE:
({candidate.subject}) -[{candidate.relation}]-> ({candidate.object})
[confidence={candidate.confidence}]

Decision policy:
- "keep_existing"
- "keep_new"
- "keep_both"
- "merge"
- "escalate"

Review rules:
- Use the source text and governance routing to decide which fact should remain.
- Do not keep a new conflicting triple just because it is plausible.
- Merge only when the merged triple is directly source-supported.

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

    def _entity_neighborhood_context(self, candidate: Any, limit: int = 12) -> str:
        """Compact 1-hop context around candidate endpoints for graph-aware review."""
        endpoint_ids = [
            str(getattr(candidate, "subject", "") or ""),
            str(getattr(candidate, "object", "") or ""),
        ]
        lines: List[str] = []
        seen = set()
        for entity_id in endpoint_ids:
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            entity = self.base_kg.entities.get(entity_id)
            if entity is not None:
                label = ", ".join(entity.labels[:3]) if entity.labels else entity.id
                lines.append(f"ENTITY {entity_id}: labels={label}; type={entity.type or 'unknown'}")
            triples = (
                self.base_kg.get_triples_by_subject(entity_id)
                + self.base_kg.get_triples_by_object(entity_id)
            )
            for triple in triples[:limit]:
                lines.append(f"  ({triple.subject}) -[{triple.relation}]-> ({triple.object})")
        return "\n".join(lines[: max(limit * 2, limit)])


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
        skip_evidence_linking: bool = False,
        skip_verification: bool = False,
        governance_review_mode: str = "triage",
        triage_confidence_threshold: float = 0.7,
        use_base_context: bool = True,
        fixed_schema: bool = False,
        reuse_corpus_schema: bool = True,
        schema_override: Optional[Dict[str, Any]] = None,
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
        self.skip_evidence_linking = skip_evidence_linking
        self.skip_verification = skip_verification
        self.governance_review_mode = governance_review_mode
        self.triage_confidence_threshold = triage_confidence_threshold
        self.use_base_context = use_base_context
        self.fixed_schema = fixed_schema
        self.reuse_corpus_schema = reuse_corpus_schema
        self.schema_override = schema_override
        self.domain_builder = DomainBuilder(self.llm_config)
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
            "extraction": {
                "use_base_context": self.use_base_context,
                "fixed_schema": self.fixed_schema,
                "reuse_corpus_schema": self.reuse_corpus_schema,
            },
            "governance": {
                "enabled": self.enable_governance,
                "review_mode": self.governance_review_mode,
                "bootstrap_assignment_stats": {},
                "new_entities_domain_assigned": 0,
                "new_triples_reviewed": 0,
                "new_triples_auto_approved": 0,
                "new_triples_approved": 0,
                "new_triples_revised": 0,
                "new_triples_rejected": 0,
                "new_triples_escalated": 0,
                "conflicts_reviewed": 0,
                "conflicts_escalated": 0,
                "triage_reasons": {},
                "decisions": [],
            },
        }

        # Step 1: Run the extraction pipeline on new documents into a
        #         *separate* working KG so we don't pollute the base yet.
        #
        # Important: by default we seed this working KG with the current
        # base graph. That makes enrichment behave more like creation:
        # new documents are processed against existing canonical entities
        # instead of against an empty graph.
        if self.use_base_context:
            working_kg = KnowledgeGraph.from_dict(self.base_kg.to_dict())
        else:
            working_kg = KnowledgeGraph()
        pipeline = DeliberativeOrchestrator(
            llm_config=self.llm_config,
            knowledge_graph=working_kg,
            enable_governance=False,
            governance_mode="audit_only",
            skip_evidence_linking=self.skip_evidence_linking,
            skip_verification=self.skip_verification,
            quality_threshold=quality_threshold,
            max_refinement_iterations=1,
            enable_self_consistency=False,  # Speed: skip broken SC
            enable_open_world=not self.fixed_schema,
            enable_cross_document=self.use_base_context,
            enable_deliberation=False,  # Speed: skip fake deliberation
            reuse_corpus_schema=self.reuse_corpus_schema,
            schema_override=self.schema_override if self.fixed_schema else None,
        )

        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Extracting from new documents")
        print("=" * 70)

        pipeline.process_corpus(documents)

        # Step 2: Compute diff between delta KG and base KG
        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Computing diff")
        print("=" * 70)

        diff = compute_diff(self.base_kg, working_kg, self.match_threshold)
        print(diff.summary())
        report["diffs"].append({
            "new_entities": len(diff.new_entities),
            "updated_entities": len(diff.updated_entities),
            "new_triples": len(diff.new_triples),
            "conflicting_triples": len(diff.conflicting_triples),
        })

        source_texts = " ".join(d.get("text", "")[:1000] for d in documents)

        # Step 2b: Mirror creation-time provisional entity assignment so
        # enrichment routes new-entity triples through the current org chart
        # before governance review.
        if diff.new_entities and self.enable_governance and self.org_chart is not None:
            entity_payload = []
            for entity in diff.new_entities:
                labels = list(entity.labels or [])
                entity_payload.append({
                    "id": entity.id,
                    "text": labels[0] if labels else entity.id,
                    "labels": labels,
                    "mentions": labels,
                    "type": entity.type or "",
                })
            provisional_assignments, bootstrap_stats = self.domain_builder.assign_entities_to_org_chart(
                entity_payload,
                self.org_chart,
                return_diagnostics=True,
            )
            assigned_count = 0
            for entity in diff.new_entities:
                domain_ids = provisional_assignments.get(entity.id, [])
                if domain_ids:
                    self.org_chart.assign_entity(entity.id, domain_ids)
                    assigned_count += 1
            report["governance"]["bootstrap_assignment_stats"] = bootstrap_stats
            report["governance"]["new_entities_domain_assigned"] = assigned_count

        # Step 3: Review new triples with the governing expert(s)
        governed_new_triple_decisions: List[GovernanceDecision] = []
        if diff.new_triples and self.governance_board:
            print(f"\nReviewing {len(diff.new_triples)} proposed triples with domain owners...")
            approved_triples = []
            triples_to_review = []
            precomputed_assignment: Dict[int, Dict[str, Any]] = {}

            for idx, candidate_t in enumerate(diff.new_triples):
                assignment = self.org_chart.route_triple_for_governance(candidate_t)
                triage_reason = self._triage_reason(candidate_t, assignment)
                precomputed_assignment[idx] = {
                    "assignment": assignment,
                    "triage_reason": triage_reason,
                }
                if self.governance_review_mode == "triage" and triage_reason is None:
                    approved_triples.append(candidate_t)
                    report["governance"]["new_triples_auto_approved"] += 1
                    governed_new_triple_decisions.append(
                        GovernanceDecision(
                            triple=candidate_t,
                            action="auto_approve",
                            domain_id=assignment.primary_domain_id,
                            rationale="Triaged enrichment auto-approved a low-risk proposal.",
                            assignment=assignment,
                        )
                    )
                    report["governance"]["decisions"].append({
                        "triple": {
                            "subject": candidate_t.subject,
                            "relation": candidate_t.relation,
                            "object": candidate_t.object,
                        },
                        "action": "auto_approve",
                        "owner_domains": assignment.domain_ids,
                        "rationale": "Triaged enrichment auto-approved a low-risk proposal.",
                    })
                    continue
                triples_to_review.append((idx, candidate_t))
                if triage_reason is not None:
                    report["governance"]["triage_reasons"][triage_reason] = (
                        report["governance"]["triage_reasons"].get(triage_reason, 0) + 1
                    )

            governance_decisions = self.governance_board.review_new_triples(
                [candidate for _, candidate in triples_to_review],
                source_texts,
            )

            for (idx, candidate_t), decision in zip(triples_to_review, governance_decisions):
                action = decision.get("action", "reject")
                triage_reason = precomputed_assignment[idx]["triage_reason"]
                assignment = precomputed_assignment[idx]["assignment"]
                report["governance"]["new_triples_reviewed"] += 1
                rationale = decision.get("rationale", "")
                if triage_reason:
                    rationale = f"{rationale} [triage_reason={triage_reason}]".strip()
                report["governance"]["decisions"].append({
                    "triple": {
                        "subject": candidate_t.subject,
                        "relation": candidate_t.relation,
                        "object": candidate_t.object,
                    },
                    "action": action,
                    "owner_domains": decision.get("owner_domains", []),
                    "rationale": rationale,
                })

                if action == "approve":
                    approved_triples.append(candidate_t)
                    report["governance"]["new_triples_approved"] += 1
                    governed_new_triple_decisions.append(
                        GovernanceDecision(
                            triple=candidate_t,
                            action="approve",
                            domain_id=assignment.primary_domain_id,
                            rationale=rationale,
                            assignment=assignment,
                        )
                    )
                elif action == "revise" and decision.get("revised_triple"):
                    revised = decision["revised_triple"]
                    revised_triple = Triple(
                        subject=revised.get("subject", candidate_t.subject),
                        relation=revised.get("relation", candidate_t.relation),
                        object=revised.get("object", candidate_t.object),
                        confidence=candidate_t.confidence,
                        source=candidate_t.source,
                        metadata=candidate_t.metadata,
                    )
                    approved_triples.append(revised_triple)
                    report["governance"]["new_triples_revised"] += 1
                    governed_new_triple_decisions.append(
                        GovernanceDecision(
                            triple=candidate_t,
                            action="revise",
                            domain_id=assignment.primary_domain_id,
                            rationale=rationale,
                            revised_triple=revised_triple,
                            assignment=assignment,
                        )
                    )
                elif action == "escalate":
                    report["governance"]["new_triples_escalated"] += 1
                    governed_new_triple_decisions.append(
                        GovernanceDecision(
                            triple=candidate_t,
                            action="escalate",
                            domain_id=assignment.primary_domain_id,
                            rationale=rationale,
                            assignment=assignment,
                        )
                    )
                else:
                    report["governance"]["new_triples_rejected"] += 1
                    governed_new_triple_decisions.append(
                        GovernanceDecision(
                            triple=candidate_t,
                            action="reject",
                            domain_id=assignment.primary_domain_id,
                            rationale=rationale,
                            assignment=assignment,
                        )
                    )

            diff.new_triples = approved_triples

        # Step 4: Resolve conflicts
        conflict_strategy = "keep_higher_confidence"
        governed_conflict_decisions: List[GovernanceDecision] = []
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
                    assignment = (
                        self.org_chart.route_triple_for_governance(candidate_t)
                        if self.org_chart is not None
                        else None
                    )
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
                        if assignment is not None:
                            governed_conflict_decisions.append(
                                GovernanceDecision(
                                    triple=candidate_t,
                                    action="approve",
                                    domain_id=assignment.primary_domain_id,
                                    rationale=res.get("rationale", ""),
                                    assignment=assignment,
                                )
                            )
                    elif resolution == "keep_both":
                        diff.new_triples.append(candidate_t)
                        resolved_triples.append(idx)
                        if assignment is not None:
                            governed_conflict_decisions.append(
                                GovernanceDecision(
                                    triple=candidate_t,
                                    action="approve",
                                    domain_id=assignment.primary_domain_id,
                                    rationale=res.get("rationale", ""),
                                    assignment=assignment,
                                )
                            )
                    elif resolution == "merge" and res.get("merged_triple"):
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
                        if assignment is not None:
                            governed_conflict_decisions.append(
                                GovernanceDecision(
                                    triple=candidate_t,
                                    action="revise",
                                    domain_id=assignment.primary_domain_id,
                                    rationale=res.get("rationale", ""),
                                    revised_triple=merged,
                                    assignment=assignment,
                                )
                            )
                    elif resolution == "escalate":
                        report["governance"]["conflicts_escalated"] += 1
                        if assignment is not None:
                            governed_conflict_decisions.append(
                                GovernanceDecision(
                                    triple=candidate_t,
                                    action="escalate",
                                    domain_id=assignment.primary_domain_id,
                                    rationale=res.get("rationale", ""),
                                    assignment=assignment,
                                )
                            )
                    # else: keep_existing, do nothing
                    if resolution == "keep_existing" and assignment is not None:
                        governed_conflict_decisions.append(
                            GovernanceDecision(
                                triple=candidate_t,
                                action="reject",
                                domain_id=assignment.primary_domain_id,
                                rationale=res.get("rationale", ""),
                                assignment=assignment,
                            )
                        )

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

        if self.governed_kg is not None and self.enable_governance:
            merge_stats = self._merge_governed_diff(
                diff,
                governed_new_triple_decisions,
                governed_conflict_decisions,
            )
        else:
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

    def _triage_reason(self, candidate: Any, assignment: Any) -> Optional[str]:
        confidence = candidate.confidence or 0.0
        if confidence < self.triage_confidence_threshold:
            return "low_confidence"
        if assignment.assignment_type in {"cross_domain", "unowned"}:
            return assignment.assignment_type
        if self._relation_outside_schema(candidate.relation):
            return "schema_novel"
        return None

    def _relation_outside_schema(self, relation: str) -> bool:
        if self.org_chart is None or not self.org_chart.domains:
            return False
        allowed_relations = set()
        for domain in self.org_chart.domains:
            allowed_relations.update(domain.relation_schema.keys())
            domain_metadata = coerce_metadata(domain.metadata)
            allowed_relations.update(domain_metadata.get("seed_relation_types", []))
        if not allowed_relations:
            return False
        if relation in allowed_relations:
            return False
        normalized = relation.upper().replace("-", "_").replace(" ", "_")
        for allowed in allowed_relations:
            if allowed.upper().replace("-", "_").replace(" ", "_") == normalized:
                return False
        return True

    def _merge_governed_diff(
        self,
        diff: KGDiff,
        new_triple_decisions: List[GovernanceDecision],
        conflict_decisions: List[GovernanceDecision],
    ) -> Dict[str, Any]:
        stats = {
            "entities_added": 0,
            "entities_updated": 0,
            "triples_added": 0,
            "conflicts_resolved": len(conflict_decisions),
        }

        for entity in diff.new_entities:
            self.base_kg.add_entity(
                entity_id=entity.id,
                labels=entity.labels,
                entity_type=entity.type,
                metadata=entity.metadata,
            )
            stats["entities_added"] += 1

        for existing, incoming in diff.updated_entities:
            entity = self.base_kg.entities[existing.id]
            for label in incoming.labels:
                if label not in entity.labels:
                    entity.labels.append(label)
            if incoming.type and not entity.type:
                entity.type = incoming.type
            entity.metadata.update(incoming.metadata)
            stats["entities_updated"] += 1

        for decision in [*new_triple_decisions, *conflict_decisions]:
            if decision.action in {"approve", "auto_approve", "revise"}:
                result = self.governed_kg.commit_decision(decision)
                if result is not None:
                    stats["triples_added"] += 1
            else:
                if decision not in self.governed_kg.audit_log:
                    self.governed_kg.audit_log.append(decision)

        self.governed_kg.org_chart.refresh_cross_domain_relations(self.base_kg)
        return stats

    def save(self, path: str) -> None:
        """Save the enriched KG to disk."""
        if self.governed_kg is not None:
            save_governed_kg(self.governed_kg, path)
        else:
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
