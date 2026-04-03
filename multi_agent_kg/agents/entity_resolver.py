"""
Entity Resolver Agent -- Global Entity Resolution (Phase 4).

Replaces per-segment coreference resolution with a global entity resolution
pass inspired by KGGen-style clustering.  The pipeline has three steps:

1. **Fast candidate pairing** (no LLM): normalise entity text and use
   rapidfuzz token-sort-ratio to find pairs whose similarity >= 0.85.
   Falls back to simple substring matching when rapidfuzz is not installed.
2. **LLM-based pair verification**: for every candidate pair the LLM decides
   (via constrained decoding with ``EntityResolutionPair`` schema) whether the
   two mentions refer to the same real-world entity.  Positive decisions are
   accumulated into clusters using a union-find structure.
3. **Canonicalisation**: each cluster gets a canonical name (most frequent
   mention, then longest), its entity type is mapped to the closest schema
   type from the domain config, and all other mentions are preserved as
   aliases.  Confidence = max confidence among cluster members.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.schemas.extraction_schemas import EntityResolutionPair

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType

# ---------------------------------------------------------------------------
# Optional dependency: rapidfuzz
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Union-Find (disjoint-set) with path compression and union by rank
# ---------------------------------------------------------------------------

class UnionFind:
    """Simple union-find / disjoint-set structure."""

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}
        self._rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def clusters(self, indices: List[int]) -> Dict[int, List[int]]:
        """Return ``{root: [members ...]}`` for the given indices."""
        groups: Dict[int, List[int]] = {}
        for idx in indices:
            root = self.find(idx)
            groups.setdefault(root, []).append(idx)
        return groups


# ---------------------------------------------------------------------------
# Pair-verification LLM prompt
# ---------------------------------------------------------------------------

PAIR_VERIFICATION_PROMPT = """You are an entity resolution expert.

Determine whether the following two entity mentions refer to the **same
real-world entity**.

Entity A: "{entity_a}" (type: {type_a})
Entity B: "{entity_b}" (type: {type_b})

Context (domain): {domain}

If they are the same entity, choose a single canonical (preferred) name.
If they are NOT the same entity, set canonical_name to the text of Entity A.

You MUST provide a reason explaining WHY these are or are not the same entity.
The reason should be 1-2 sentences describing the evidence. Never leave reason empty.

Respond with JSON matching the schema exactly."""


# ---------------------------------------------------------------------------
# EntityResolver agent
# ---------------------------------------------------------------------------

class EntityResolver(BaseAgent):
    """
    Global Entity Resolution agent (Phase 4).

    Accepts all extracted entities across every segment and produces a
    deduplicated set with canonical names, merged aliases, and per-entity
    confidence scores derived from the cluster members (max, not hardcoded).
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        fuzzy_threshold: float = 0.75,
    ):
        super().__init__(
            name="EntityResolver",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Run global entity resolution over all extracted entities.

        Args:
            context: Processing context (document id, domain, etc.).
            entities: Flat list of entity dicts from all segments. Each dict
                is expected to have at least ``text`` and ``type`` keys,
                and optionally ``confidence``, ``aliases``, ``mentions``,
                ``source_segment``, etc.
            domain_config: Domain configuration; may contain
                ``entity_types`` (list of schema type strings/dicts).

        Returns:
            ExtractionResult whose ``items`` are the resolved, deduplicated
            entities.
        """
        self.stats["calls"] += 1
        entities = entities or []

        # Edge cases: nothing to resolve
        if not entities:
            self.log("No entities to resolve -- returning empty result.")
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={"document_id": context.document_id, "clusters": 0},
            )

        if len(entities) == 1:
            self.log("Single entity -- nothing to cluster.")
            entity = entities[0]
            entity.setdefault("aliases", [])
            return ExtractionResult(
                items=[entity],
                confidence=entity.get("confidence", 0.5),
                metadata={"document_id": context.document_id, "clusters": 1},
            )

        # Derive schema types from domain config for type-mapping step
        schema_types = self._extract_schema_types(domain_config)

        # Step 1 -- fast fuzzy candidate pairing (no LLM)
        candidate_pairs = self._step1_candidate_pairs(entities)
        self.log(
            f"Step 1: {len(candidate_pairs)} candidate pairs from "
            f"{len(entities)} entities."
        )

        # Step 2 -- LLM pair verification + union-find clustering
        uf, merge_log = self._step2_llm_clustering(entities, candidate_pairs, context)
        self.log(f"Step 2: {len(merge_log)} merges performed.")

        # Step 3 -- canonicalisation
        resolved = self._step3_canonicalise(entities, uf, schema_types)
        self.log(f"Step 3: {len(resolved)} resolved entities after clustering.")

        # Attach merge_log to each resolved entity that has aliases
        for entity in resolved:
            entity_names = {entity["text"]} | set(entity.get("aliases", []))
            entity["merge_pairs"] = [
                m for m in merge_log
                if m["entity_a"] in entity_names or m["entity_b"] in entity_names
            ]

        # Store in shared memory
        if self.shared_memory:
            self._store_resolved_entities(resolved, context.document_id)
            # Also persist the merge log for future reference
            self.store_in_memory(
                memory_type=MemoryType.PROCEDURAL,
                content={
                    "merge_log": merge_log,
                    "document_id": context.document_id,
                },
            )

        # Overall confidence = mean of individual resolved confidences
        if resolved:
            avg_conf = sum(e.get("confidence", 0.5) for e in resolved) / len(resolved)
        else:
            avg_conf = 0.0

        return ExtractionResult(
            items=resolved,
            confidence=avg_conf,
            metadata={
                "document_id": context.document_id,
                "input_entity_count": len(entities),
                "resolved_entity_count": len(resolved),
                "clusters": len(resolved),
                "candidate_pairs_evaluated": len(candidate_pairs),
            },
        )

    # ------------------------------------------------------------------ #
    #  Step 1 -- fast fuzzy candidate pairing                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Lower-case, strip accents, collapse whitespace."""
        text = text.strip().lower()
        # Strip combining characters (accents)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _simple_fuzzy_match(a: str, b: str) -> float:
        """
        Fallback fuzzy similarity when rapidfuzz is unavailable.

        Combines exact-match, substring containment, and token-overlap
        into a score in [0, 1].
        """
        if a == b:
            return 1.0

        # Substring containment check
        if a in b or b in a:
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            return len(shorter) / len(longer)

        # Token overlap (Jaccard-ish)
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _strip_suffixes(text: str) -> str:
        """Remove common corporate/organisational suffixes for token comparison."""
        _NOISE = {"inc", "inc.", "ltd", "ltd.", "llc", "co", "co.", "corp", "corp.", "the"}
        tokens = text.split()
        return " ".join(t for t in tokens if t not in _NOISE)

    def _step1_candidate_pairs(
        self,
        entities: List[Dict[str, Any]],
    ) -> List[Tuple[int, int]]:
        """Return index pairs whose normalised text similarity >= threshold.

        Three heuristics fire *before* the fuzzy-score check so that
        semantically related entities that differ in surface form are still
        surfaced as candidates for LLM verification:

        1. **Containment**: the shorter normalised text is fully contained in
           the longer one.
        2. **Shared-type + shared-token**: both entities share the same type
           AND at least one significant token (length >= 3 characters).
        3. **Acronym / first-word**: one entity is a single token that matches
           the first token of the other entity, and they share the same type.
        """
        n = len(entities)
        normed = [self._normalise_text(e.get("text", "")) for e in entities]
        types = [
            self._normalise_text(e.get("type", "")) for e in entities
        ]
        # Pre-compute significant token sets (tokens with len >= 3) with
        # noise suffixes stripped so "inc", "ltd", etc. don't cause spurious
        # matches.
        sig_tokens = [
            {t for t in self._strip_suffixes(nm).split() if len(t) >= 3}
            for nm in normed
        ]
        pairs: List[Tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if not normed[i] or not normed[j]:
                    continue

                # Exact normalised match is an automatic candidate
                if normed[i] == normed[j]:
                    pairs.append((i, j))
                    continue

                # --- Heuristic 1: containment check ---
                shorter, longer = (
                    (normed[i], normed[j])
                    if len(normed[i]) <= len(normed[j])
                    else (normed[j], normed[i])
                )
                if shorter and shorter in longer:
                    if len(shorter) / len(longer) >= 0.5:
                        pairs.append((i, j))
                        continue

                # --- Heuristic 2: shared type + shared significant token ---
                same_type = types[i] and types[j] and types[i] == types[j]
                shared = sig_tokens[i] & sig_tokens[j]
                if same_type and shared:
                    # Products need 2+ shared tokens to prevent over-merging
                    # (e.g., "iMac" and "MacBook" share only "mac" → should NOT pair)
                    is_product = "product" in types[i].lower()
                    min_shared = 2 if is_product else 1
                    if len(shared) >= min_shared:
                        pairs.append((i, j))
                        continue

                # --- Heuristic 3: acronym / first-word match (same type) ---
                if same_type:
                    tokens_i = normed[i].split()
                    tokens_j = normed[j].split()
                    if (
                        len(tokens_i) == 1
                        and tokens_j
                        and tokens_i[0] == tokens_j[0]
                    ):
                        pairs.append((i, j))
                        continue
                    if (
                        len(tokens_j) == 1
                        and tokens_i
                        and tokens_j[0] == tokens_i[0]
                    ):
                        pairs.append((i, j))
                        continue

                # --- Fuzzy similarity (original gate) ---
                if RAPIDFUZZ_AVAILABLE:
                    score = fuzz.token_sort_ratio(normed[i], normed[j]) / 100.0
                else:
                    score = self._simple_fuzzy_match(normed[i], normed[j])
                if score >= self.fuzzy_threshold:
                    pairs.append((i, j))

        return pairs

    # ------------------------------------------------------------------ #
    #  Step 2 -- LLM pair verification + union-find                        #
    # ------------------------------------------------------------------ #

    def _step2_llm_clustering(
        self,
        entities: List[Dict[str, Any]],
        candidate_pairs: List[Tuple[int, int]],
        context: AgentContext,
    ) -> Tuple[UnionFind, List[Dict[str, Any]]]:
        """Verify each candidate pair with the LLM and build clusters.

        Returns:
            Tuple of (union_find, merge_log) where merge_log records every
            accepted merge with the pair texts and the LLM's reason.
        """
        uf = UnionFind()
        merge_log: List[Dict[str, Any]] = []

        # Ensure every entity index is registered (singletons too)
        for idx in range(len(entities)):
            uf.find(idx)

        if not candidate_pairs:
            return uf, merge_log

        domain = context.domain or "General"

        for idx_a, idx_b in candidate_pairs:
            entity_a = entities[idx_a]
            entity_b = entities[idx_b]

            prompt = PAIR_VERIFICATION_PROMPT.format(
                entity_a=entity_a.get("text", ""),
                type_a=entity_a.get("type", "UNKNOWN"),
                entity_b=entity_b.get("text", ""),
                type_b=entity_b.get("type", "UNKNOWN"),
                domain=domain,
            )

            try:
                result = self.call_llm(
                    prompt=prompt,
                    system_prompt=(
                        "You are an entity resolution expert. Decide whether "
                        "two entity mentions refer to the same real-world entity."
                    ),
                    tier=ModelTier.MEDIUM,
                    temperature=0.1,
                    response_schema=EntityResolutionPair,
                )

                same = result.get("same_entity", False)
                reason = result.get("reason", "")
                canonical = result.get("canonical_name", "")

                if same:
                    uf.union(idx_a, idx_b)
                    merge_record = {
                        "entity_a": entity_a.get("text", ""),
                        "entity_b": entity_b.get("text", ""),
                        "canonical_name": canonical,
                        "reason": reason,
                        "method": "llm",
                    }
                    merge_log.append(merge_record)
                    self.log(
                        f"  Merged: \"{entity_a.get('text')}\" <-> "
                        f"\"{entity_b.get('text')}\" "
                        f"(reason: {reason})"
                    )
            except Exception as exc:
                self.log(
                    f"LLM pair check failed for "
                    f"\"{entity_a.get('text')}\" / \"{entity_b.get('text')}\": "
                    f"{exc}",
                    level="WARNING",
                )
                # On failure, fall back to surface-form heuristic: merge if
                # normalised forms are identical.
                norm_a = self._normalise_text(entity_a.get("text", ""))
                norm_b = self._normalise_text(entity_b.get("text", ""))
                if norm_a and norm_a == norm_b:
                    uf.union(idx_a, idx_b)
                    merge_log.append({
                        "entity_a": entity_a.get("text", ""),
                        "entity_b": entity_b.get("text", ""),
                        "canonical_name": entity_a.get("text", ""),
                        "reason": "Identical normalised forms (LLM fallback)",
                        "method": "heuristic",
                    })

        return uf, merge_log

    # ------------------------------------------------------------------ #
    #  Step 3 -- canonicalisation                                          #
    # ------------------------------------------------------------------ #

    def _step3_canonicalise(
        self,
        entities: List[Dict[str, Any]],
        uf: UnionFind,
        schema_types: List[str],
    ) -> List[Dict[str, Any]]:
        """
        For each cluster, select a canonical name, map the type to the
        closest schema type, and preserve all aliases.
        """
        indices = list(range(len(entities)))
        cluster_map = uf.clusters(indices)
        resolved: List[Dict[str, Any]] = []

        for _root, members in cluster_map.items():
            member_entities = [entities[m] for m in members]

            # --- canonical name: prefer formal proper names ---
            mention_texts = [e.get("text", "") for e in member_entities]
            canonical_name = self._pick_canonical_name(mention_texts)

            # --- aliases: every unique mention that is not the canonical ---
            all_mentions: List[str] = []
            for e in member_entities:
                text = e.get("text", "")
                if text and text not in all_mentions:
                    all_mentions.append(text)
                for alias in e.get("aliases", []):
                    if alias and alias not in all_mentions:
                        all_mentions.append(alias)
                for mention in e.get("mentions", []):
                    if mention and mention not in all_mentions:
                        all_mentions.append(mention)
            aliases = [m for m in all_mentions if m != canonical_name]

            # --- entity type: most common type in cluster, mapped to schema ---
            type_counts = Counter(
                e.get("type", "UNKNOWN") for e in member_entities
            )
            raw_type = type_counts.most_common(1)[0][0]
            mapped_type = self._map_to_schema_type(raw_type, schema_types)

            # --- confidence: max among cluster members (NOT hardcoded) ---
            confidence = max(
                e.get("confidence", 0.0) for e in member_entities
            )

            # --- preserve extra metadata from the highest-confidence member ---
            best_member = max(
                member_entities, key=lambda e: e.get("confidence", 0.0)
            )
            source_segment = best_member.get("source_segment")

            resolved_entity: Dict[str, Any] = {
                "text": canonical_name,
                "type": mapped_type,
                "confidence": confidence,
                "aliases": aliases,
                "mentions": all_mentions,
                "cluster_size": len(members),
                "source_segment": source_segment,
            }

            # Propagate id if one exists in the cluster
            existing_ids = [
                e.get("id") for e in member_entities if e.get("id")
            ]
            if existing_ids:
                resolved_entity["id"] = existing_ids[0]

            resolved.append(resolved_entity)

            # Register aliases in shared memory
            if self.shared_memory and aliases:
                canonical_id = resolved_entity.get("id", canonical_name)
                for alias in aliases:
                    try:
                        self.shared_memory.register_entity_alias(
                            alias, canonical_id
                        )
                    except Exception:
                        pass  # best-effort

        return resolved

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_canonical_name(mentions: List[str]) -> str:
        """Choose the best canonical name from a cluster of mentions.

        Priority order:
        1. Formal names containing corporate suffixes (Inc., Corp., Ltd.)
        2. Full proper names for people (contains space, e.g. "Steve Jobs" > "Jobs")
        3. Most frequent mention
        4. Longest name among equally frequent (but NOT phrases with generic
           words like "retail chain", "labor conditions", etc.)

        This prevents "Apple Store retail chain" from winning over "Apple Inc."
        and "Jobs" from winning over "Steve Jobs".
        """
        if not mentions:
            return ""
        if len(mentions) == 1:
            return mentions[0]

        freq = Counter(mentions)

        # Tier 1: formal corporate suffixes
        _corp_suffixes = ("Inc.", "Inc", "Corp.", "Corp", "Ltd.", "Ltd", "LLC",
                          "Co.", "Company", "Corporation")
        formal = [m for m in mentions if any(m.endswith(s) or f" {s}" in m for s in _corp_suffixes)]
        if formal:
            return max(formal, key=lambda m: (freq[m], len(m)))

        # Tier 2: full person names (multi-word, starts uppercase, no generic filler)
        _generic_words = {"retail", "chain", "labor", "conditions", "practices",
                          "ethics", "tactics", "sourcing", "smaller", "businesses",
                          "valuable", "brands", "capitalization", "revenue"}
        proper_names = [
            m for m in mentions
            if " " in m
            and m[0].isupper()
            and not any(w.lower() in _generic_words for w in m.split())
        ]
        if proper_names:
            return max(proper_names, key=lambda m: (freq[m], len(m)))

        # Tier 3: most frequent, then longest (but filter out very long phrases)
        max_freq = max(freq.values())
        most_frequent = [t for t, c in freq.items() if c == max_freq]
        # Prefer shorter proper names over long descriptions
        candidates = [m for m in most_frequent if len(m) <= 40]
        if not candidates:
            candidates = most_frequent
        return max(candidates, key=len)

    @staticmethod
    def _extract_schema_types(
        domain_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Pull flat list of UPPER_SNAKE_CASE type names from domain config."""
        if not domain_config:
            return []
        raw = domain_config.get("entity_types", [])
        types: List[str] = []
        for item in raw:
            if isinstance(item, dict):
                types.append(item.get("type", ""))
            elif isinstance(item, str):
                types.append(item)
        return [t for t in types if t]

    def _map_to_schema_type(self, raw_type: str, schema_types: List[str]) -> str:
        """
        Map *raw_type* to the closest type in *schema_types*.

        Uses normalised exact match first, then token-overlap, and falls back
        to returning *raw_type* unchanged if no good match is found.
        """
        if not schema_types:
            return raw_type

        norm_raw = self._normalise_text(raw_type.replace("_", " "))

        # Pass 1 -- exact normalised match
        for st in schema_types:
            if self._normalise_text(st.replace("_", " ")) == norm_raw:
                return st

        # Pass 2 -- best fuzzy match
        best_score = 0.0
        best_type = raw_type
        for st in schema_types:
            norm_st = self._normalise_text(st.replace("_", " "))
            if RAPIDFUZZ_AVAILABLE:
                score = fuzz.token_sort_ratio(norm_raw, norm_st) / 100.0
            else:
                score = self._simple_fuzzy_match(norm_raw, norm_st)
            if score > best_score:
                best_score = score
                best_type = st

        # Only accept if reasonably close
        if best_score >= 0.6:
            return best_type
        return raw_type

    def _store_resolved_entities(
        self,
        entities: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Persist resolved entities to shared memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "resolved_entities": entities,
                "document_id": document_id,
            },
        )

        for entity in entities:
            try:
                self.shared_memory.add_entity_context(
                    entity_id=entity.get("id", entity.get("text", "")),
                    context={
                        "document_id": document_id,
                        "type": entity.get("type"),
                        "confidence": entity.get("confidence"),
                        "aliases": entity.get("aliases", []),
                        "cluster_size": entity.get("cluster_size", 1),
                    },
                )
            except Exception:
                pass  # best-effort

    # ------------------------------------------------------------------ #
    #  Deliberation / voting interface                                     #
    # ------------------------------------------------------------------ #

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple["VoteType", float, str]:
        """EntityResolver voting logic for deliberation."""
        from multi_agent_kg.core.deliberation import VoteType

        if hypothesis_type == "entity":
            return self._vote_on_entity(hypothesis_content)
        return VoteType.ABSTAIN, 0.5, "EntityResolver cannot evaluate this hypothesis type"

    def _vote_on_entity(
        self,
        entity: Dict[str, Any],
    ) -> Tuple["VoteType", float, str]:
        """Vote on an entity hypothesis based on resolution heuristics."""
        from multi_agent_kg.core.deliberation import VoteType

        text = entity.get("text", "")
        if not text or len(text) < 2:
            return VoteType.REJECT, 0.9, "Entity text too short or empty"
        if len(text) > 200:
            return VoteType.WEAK_REJECT, 0.7, "Entity text suspiciously long"
        return VoteType.WEAK_ACCEPT, 0.6, "Entity passes basic resolution checks"
