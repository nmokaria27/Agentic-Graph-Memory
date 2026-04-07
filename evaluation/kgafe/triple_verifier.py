"""
Three-Tier Triple Verifier — verifies atomic facts against the knowledge graph.

Verification tiers (in order of rigor):
  Tier 1: Exact Match — fact maps directly to a KG triple
  Tier 2: Path-Based — fact is entailed by a multi-hop path through the KG
  Tier 3: Semantic Entailment — LLM judges whether KG neighborhood entails the fact

Each tier returns a VerificationResult with a verdict, supporting evidence,
and a confidence score. The framework uses the STRICTEST tier that produces
a confident verdict (exact > path > semantic).

Novel contribution: unlike FActScore (Wikipedia) or SAFE (web search), this
verifies against a structured graph, enabling precise path-based entailment
that text-based approaches cannot achieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.domain_experts import find_paths, neighbourhood
from multi_agent_kg.core.kg_operations import normalize_entity_name, normalize_for_matching
from multi_agent_kg.llm.openai_client import chat_completion_json


class Verdict(Enum):
    """Verification verdict for an atomic fact."""
    SUPPORTED = "supported"  # Fact is supported by KG evidence
    CONTRADICTED = "contradicted"  # Fact contradicts KG evidence
    UNVERIFIABLE = "unverifiable"  # KG has no relevant information
    PARTIALLY_SUPPORTED = "partially_supported"  # Some aspects supported, others not


@dataclass
class VerificationResult:
    """Result of verifying a single atomic fact against the KG."""

    fact_id: str
    verdict: Verdict
    tier: int  # 1=exact, 2=path, 3=semantic
    confidence: float  # 0.0-1.0
    supporting_triples: List[Triple] = field(default_factory=list)
    supporting_paths: List[List[Triple]] = field(default_factory=list)
    reasoning: str = ""
    matched_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "verdict": self.verdict.value,
            "tier": self.tier,
            "confidence": self.confidence,
            "supporting_triples": [
                f"({t.subject}) -[{t.relation}]-> ({t.object})"
                for t in self.supporting_triples
            ],
            "supporting_paths": [
                [f"({t.subject}) -[{t.relation}]-> ({t.object})" for t in path]
                for path in self.supporting_paths
            ],
            "reasoning": self.reasoning,
            "matched_entities": self.matched_entities,
        }


class TripleVerifier:
    """
    Three-tier verification of atomic facts against a knowledge graph.

    Tier 1: Exact triple match (highest precision)
    Tier 2: Multi-hop path inference (structural entailment)
    Tier 3: LLM semantic entailment (broadest coverage)
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "gemma3:27b",
        path_max_hops: int = 3,
        neighbourhood_hops: int = 2,
    ):
        self.kg = kg
        self.model = model
        self.path_max_hops = path_max_hops
        self.neighbourhood_hops = neighbourhood_hops

        # Pre-build indexes for fast lookup
        self._entity_name_to_id = self._build_entity_index()
        self._relation_index = self._build_relation_index()

    def _build_entity_index(self) -> Dict[str, List[str]]:
        """Map normalized entity names/labels to entity IDs.

        Uses both standard normalization (for backward compat) and
        aggressive normalization (strips all non-alphanumeric) so that
        entity IDs like ``homair`` match display names like ``HOMA-IR``.
        Also indexes names that appear as triple subjects/objects, since
        triples may use display names rather than entity IDs.
        """
        index: Dict[str, List[str]] = {}

        # Phase 1: index entity IDs and their labels
        for eid, entity in self.kg.entities.items():
            names = [eid, eid.replace("_", " ")] + entity.labels
            for name in names:
                # Standard normalization
                key = normalize_entity_name(name)
                if key:
                    index.setdefault(key, []).append(eid)
                # Aggressive normalization
                agg_key = normalize_for_matching(name)
                if agg_key and agg_key != key:
                    index.setdefault(agg_key, []).append(eid)

        # Phase 2: index names as they appear in triple subjects/objects.
        # This handles cases where triples store display names that don't
        # exactly match any entity ID or label after normalization.
        triple_name_to_entity: Dict[str, str] = {}
        for t in self.kg.triples:
            for name in (t.subject, t.object):
                agg = normalize_for_matching(name)
                if agg not in triple_name_to_entity:
                    triple_name_to_entity[agg] = name

        for agg_key, raw_name in triple_name_to_entity.items():
            if agg_key not in index:
                # No entity ID maps here — find the closest entity
                std_key = normalize_entity_name(raw_name)
                if std_key in index:
                    index[agg_key] = index[std_key]

        # Deduplicate entity ID lists
        for key in index:
            index[key] = list(dict.fromkeys(index[key]))

        return index

    def _build_relation_index(self) -> Dict[str, List[Triple]]:
        """Index triples by normalized relation name."""
        index: Dict[str, List[Triple]] = {}
        for t in self.kg.triples:
            key = t.relation.lower().replace("_", " ").strip()
            index.setdefault(key, []).append(t)
        return index

    def _resolve_entity(self, name: str) -> List[str]:
        """Resolve an entity name to KG entity IDs.

        Tries, in order:
        1. Standard normalization exact match
        2. Aggressive normalization exact match (catches ID-vs-label mismatches)
        3. Substring match on standard normalized forms
        """
        norm = normalize_entity_name(name)

        # 1. Exact standard match
        if norm in self._entity_name_to_id:
            return self._entity_name_to_id[norm]

        # 2. Exact aggressive match
        agg = normalize_for_matching(name)
        if agg in self._entity_name_to_id:
            return self._entity_name_to_id[agg]

        # 3. Substring match on standard normalized forms
        matches = []
        for key, eids in self._entity_name_to_id.items():
            if len(norm) >= 3 and (norm in key or key in norm):
                matches.extend(eids)

        return list(set(matches))

    def verify(
        self,
        fact_id: str,
        fact_text: str,
        entities_mentioned: List[str],
        relation_implied: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a single atomic fact against the KG using three tiers.

        Args:
            fact_id: Unique identifier for the fact
            fact_text: The atomic fact in natural language
            entities_mentioned: Entity names mentioned in the fact
            relation_implied: The implied relation (if identified during decomposition)

        Returns:
            VerificationResult with verdict, tier, and evidence
        """
        # Resolve entity mentions to KG IDs
        resolved_entities: Dict[str, List[str]] = {}
        for name in entities_mentioned:
            eids = self._resolve_entity(name)
            if eids:
                resolved_entities[name] = eids

        all_matched_ids = [eid for eids in resolved_entities.values() for eid in eids]

        # Tier 1: Exact triple match
        tier1_result = self._tier1_exact_match(
            fact_id, fact_text, resolved_entities, relation_implied
        )
        if tier1_result.verdict == Verdict.SUPPORTED:
            tier1_result.matched_entities = all_matched_ids
            return tier1_result

        # Tier 2: Path-based inference
        tier2_result = self._tier2_path_inference(
            fact_id, fact_text, resolved_entities
        )
        if tier2_result.verdict == Verdict.SUPPORTED:
            tier2_result.matched_entities = all_matched_ids
            return tier2_result

        # Tier 3: LLM semantic entailment
        tier3_result = self._tier3_semantic_entailment(
            fact_id, fact_text, resolved_entities
        )
        tier3_result.matched_entities = all_matched_ids
        return tier3_result

    def _tier1_exact_match(
        self,
        fact_id: str,
        fact_text: str,
        resolved_entities: Dict[str, List[str]],
        relation_implied: Optional[str],
    ) -> VerificationResult:
        """
        Tier 1: Check if the fact directly corresponds to a KG triple.

        Uses aggressive normalization so entity IDs (``homair``) match
        triple display names (``HOMA-IR``).
        """
        entity_id_lists = list(resolved_entities.values())
        if len(entity_id_lists) < 2:
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.UNVERIFIABLE,
                tier=1,
                confidence=0.0,
                reasoning="Need at least 2 resolved entities for exact triple match",
            )

        # Collect all normalised forms for each entity group so we can
        # match against triple subjects/objects (which use display names).
        entity_norm_sets: List[set] = []
        for eids in entity_id_lists:
            norms = set()
            for eid in eids:
                norms.add(normalize_for_matching(eid))
                # Also add label normalisations for this entity
                if eid in self.kg.entities:
                    for label in self.kg.entities[eid].labels:
                        norms.add(normalize_for_matching(label))
            entity_norm_sets.append(norms)

        supporting_triples = []

        for i, norms_a in enumerate(entity_norm_sets):
            for j, norms_b in enumerate(entity_norm_sets):
                if i == j:
                    continue
                for t in self.kg.triples:
                    subj_n = normalize_for_matching(t.subject)
                    obj_n = normalize_for_matching(t.object)
                    if (subj_n in norms_a and obj_n in norms_b) or \
                       (subj_n in norms_b and obj_n in norms_a):
                        if relation_implied:
                            rel_norm = normalize_for_matching(t.relation)
                            impl_norm = normalize_for_matching(relation_implied)
                            if impl_norm in rel_norm or rel_norm in impl_norm:
                                if t not in supporting_triples:
                                    supporting_triples.append(t)
                        else:
                            if t not in supporting_triples:
                                supporting_triples.append(t)

        if supporting_triples:
            avg_confidence = sum(
                t.confidence for t in supporting_triples if t.confidence
            ) / max(len([t for t in supporting_triples if t.confidence]), 1)
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.SUPPORTED,
                tier=1,
                confidence=max(0.8, avg_confidence),
                supporting_triples=supporting_triples,
                reasoning=f"Exact match: {len(supporting_triples)} triple(s) directly support this fact",
            )

        return VerificationResult(
            fact_id=fact_id,
            verdict=Verdict.UNVERIFIABLE,
            tier=1,
            confidence=0.0,
            reasoning="No exact triple match found",
        )

    def _tier2_path_inference(
        self,
        fact_id: str,
        fact_text: str,
        resolved_entities: Dict[str, List[str]],
    ) -> VerificationResult:
        """
        Tier 2: Check if a multi-hop path through the KG entails the fact.

        Finds paths between mentioned entities (up to max_hops) and checks
        if the path semantically supports the claim.
        """
        entity_id_lists = list(resolved_entities.values())
        if len(entity_id_lists) < 2:
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.UNVERIFIABLE,
                tier=2,
                confidence=0.0,
                reasoning="Need at least 2 resolved entities for path inference",
            )

        all_paths: List[List[Triple]] = []

        for i, eids_a in enumerate(entity_id_lists):
            for j, eids_b in enumerate(entity_id_lists):
                if i >= j:
                    continue
                for eid_a in eids_a[:2]:  # Limit to avoid combinatorial explosion
                    for eid_b in eids_b[:2]:
                        paths = find_paths(
                            self.kg, eid_a, eid_b, max_hops=self.path_max_hops
                        )
                        all_paths.extend(paths)

        if not all_paths:
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.UNVERIFIABLE,
                tier=2,
                confidence=0.0,
                reasoning="No multi-hop paths found between mentioned entities",
            )

        # Use LLM to check if any path entails the fact
        paths_text = []
        for k, path in enumerate(all_paths[:5]):  # Limit to 5 shortest paths
            steps = " → ".join(
                f"({t.subject}) -[{t.relation}]-> ({t.object})" for t in path
            )
            paths_text.append(f"  Path {k+1}: {steps}")

        prompt = f"""Given the following paths through a knowledge graph, determine whether
they support, contradict, or are irrelevant to the stated fact.

FACT: {fact_text}

KNOWLEDGE GRAPH PATHS:
{chr(10).join(paths_text)}

Does the chain of relationships in any path logically entail the fact?
Consider transitive relationships (e.g., A causes B, B causes C → A indirectly causes C).

Return JSON:
{{
    "verdict": "supported" | "contradicted" | "unverifiable",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of how the path(s) relate to the fact",
    "best_path_index": 1
}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a logical entailment checker. Determine if knowledge graph "
                            "paths support a given fact. Be strict — only mark as 'supported' if "
                            "the path provides genuine evidence. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.1,
            )
        except Exception as exc:
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.UNVERIFIABLE,
                tier=2,
                confidence=0.0,
                supporting_triples=[t for path in all_paths[:3] for t in path],
                reasoning=f"Tier 2 path inference failed: {exc}",
            )

        verdict_str = result.get("verdict", "unverifiable").lower()
        verdict_map = {
            "supported": Verdict.SUPPORTED,
            "contradicted": Verdict.CONTRADICTED,
            "unverifiable": Verdict.UNVERIFIABLE,
            "partially_supported": Verdict.PARTIALLY_SUPPORTED,
        }

        best_idx = self._coerce_best_path_index(result.get("best_path_index"), len(all_paths))
        best_path = all_paths[best_idx] if 0 <= best_idx < len(all_paths) else []

        return VerificationResult(
            fact_id=fact_id,
            verdict=verdict_map.get(verdict_str, Verdict.UNVERIFIABLE),
            tier=2,
            confidence=result.get("confidence", 0.5),
            supporting_triples=[t for path in all_paths[:3] for t in path],
            supporting_paths=[best_path] if best_path else [],
            reasoning=result.get("reasoning", ""),
        )

    def _coerce_best_path_index(self, raw_index: Any, num_paths: int) -> int:
        """Convert an LLM-returned path index into a safe 0-based index."""
        if num_paths <= 0:
            return -1

        if raw_index is None:
            return 0

        try:
            idx = int(raw_index)
        except (TypeError, ValueError):
            return 0

        idx -= 1  # model returns 1-based index
        if idx < 0:
            return 0
        if idx >= num_paths:
            return num_paths - 1
        return idx

    def _tier3_semantic_entailment(
        self,
        fact_id: str,
        fact_text: str,
        resolved_entities: Dict[str, List[str]],
    ) -> VerificationResult:
        """
        Tier 3: Use LLM to judge whether the KG neighborhood semantically entails the fact.

        Gathers the 2-hop neighbourhood of all mentioned entities and asks the LLM
        to judge entailment. This is the broadest but least precise tier.
        """
        # Gather neighbourhood context
        all_triples: List[Triple] = []
        for eids in resolved_entities.values():
            for eid in eids[:2]:
                nbr = neighbourhood(self.kg, eid, hops=self.neighbourhood_hops)
                all_triples.extend(nbr)

        # Deduplicate
        seen = set()
        unique_triples = []
        for t in all_triples:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        if not unique_triples:
            # Last resort: no entities resolved at all
            return VerificationResult(
                fact_id=fact_id,
                verdict=Verdict.UNVERIFIABLE,
                tier=3,
                confidence=0.0,
                reasoning="No KG entities could be resolved from the fact",
            )

        # Format triples for the LLM
        triple_lines = [
            f"  ({t.subject}) -[{t.relation}]-> ({t.object})"
            for t in unique_triples[:60]
        ]
        triples_text = "\n".join(triple_lines)

        prompt = f"""You are a fact verification judge. Given a set of knowledge graph triples
(the evidence) and a factual claim, determine whether the evidence supports,
contradicts, or cannot verify the claim.

CLAIM TO VERIFY:
{fact_text}

KNOWLEDGE GRAPH EVIDENCE (triples from the neighbourhood of mentioned entities):
{triples_text}

RULES:
- "supported": The triples collectively provide sufficient evidence for the claim.
  The claim must be directly derivable from the triples, possibly combining 2-3.
- "contradicted": The triples provide evidence that directly opposes the claim.
- "partially_supported": Some aspects of the claim are supported, others are not.
- "unverifiable": The triples don't contain enough relevant information.

Return JSON:
{{
    "verdict": "supported" | "contradicted" | "partially_supported" | "unverifiable",
    "confidence": 0.0-1.0,
    "reasoning": "Explain which triples support or contradict the claim",
    "key_triples": ["list the most relevant triples as strings"]
}}

Return ONLY the JSON."""

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact verification judge. Only mark claims as "
                        "'supported' when the evidence genuinely entails them. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )

        verdict_str = result.get("verdict", "unverifiable").lower()
        verdict_map = {
            "supported": Verdict.SUPPORTED,
            "contradicted": Verdict.CONTRADICTED,
            "unverifiable": Verdict.UNVERIFIABLE,
            "partially_supported": Verdict.PARTIALLY_SUPPORTED,
        }

        return VerificationResult(
            fact_id=fact_id,
            verdict=verdict_map.get(verdict_str, Verdict.UNVERIFIABLE),
            tier=3,
            confidence=result.get("confidence", 0.5) * 0.9,  # Slight discount for Tier 3
            supporting_triples=unique_triples[:10],
            reasoning=result.get("reasoning", ""),
        )
