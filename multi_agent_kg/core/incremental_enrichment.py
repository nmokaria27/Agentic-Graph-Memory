"""
Incremental Knowledge Graph Enrichment Pipeline.

Given an *existing* KG and one or more new documents, this module:
1. Runs extraction on the new documents (reusing the existing pipeline)
2. Computes a diff between the new extractions and the existing KG
3. Uses an LLM-backed ConflictResolver agent to adjudicate conflicts
4. Merges the accepted changes into the base KG
5. Returns a structured report of what changed

This is the multi-agent "additive" pathway — the counterpart to the
initial build pipeline in deliberative_orchestrator.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.config import LLMConfig
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
        base_kg: KnowledgeGraph,
        llm_config: Optional[LLMConfig] = None,
        match_threshold: float = 0.80,
        auto_resolve_conflicts: bool = True,
    ):
        self.base_kg = base_kg
        self.llm_config = llm_config or LLMConfig(model="gemma3:27b")
        self.match_threshold = match_threshold
        self.auto_resolve_conflicts = auto_resolve_conflicts
        self.conflict_resolver = ConflictResolver(self.llm_config)
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
        }

        # Step 1: Run the extraction pipeline on new documents into a
        #         *separate* KG so we don't pollute the base yet.
        delta_kg = KnowledgeGraph()
        pipeline = DeliberativeOrchestrator(
            llm_config=self.llm_config,
            knowledge_graph=delta_kg,
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

        # Step 3: Resolve conflicts
        conflict_strategy = "keep_higher_confidence"
        if diff.conflicting_triples and self.auto_resolve_conflicts:
            print(f"\nResolving {len(diff.conflicting_triples)} conflicts with LLM...")
            source_texts = " ".join(d.get("text", "")[:1000] for d in documents)
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
                    # else: keep_existing, do nothing

            # Remove resolved conflicts
            diff.conflicting_triples = [
                c
                for i, c in enumerate(diff.conflicting_triples)
                if i not in resolved_triples
            ]
            report["conflicts_resolved"] = len(resolved_triples)

        # Step 4: Merge
        print("\n" + "=" * 70)
        print("INCREMENTAL ENRICHMENT: Merging into base KG")
        print("=" * 70)

        merge_stats = merge_kg(self.base_kg, diff, conflict_strategy)
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
