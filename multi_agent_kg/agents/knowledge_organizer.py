"""
Knowledge Organizer Agent (Coordinator).

Responsible for:
- Final integration of verified extractions into knowledge graph
- Entity deduplication and merging
- Relation normalization
- Knowledge graph maintenance
- Export and statistics

This is the final coordinator - the output stage.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json
from collections import defaultdict
from math import sqrt

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.governance import coerce_metadata
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig, RetrievalConfig


ENTITY_DEDUP_PROMPT = """Identify duplicate entities that should be merged.

ENTITIES:
{entities_json}

Look for:
1. Same entity with different names/aliases
2. Entities that are clearly the same real-world thing
3. Abbreviations and their full forms

Return:
{{
    "merge_groups": [
        {{
            "canonical_id": "<id to keep>",
            "canonical_name": "<best name>",
            "merge_ids": ["<id1>", "<id2>", ...],
            "reason": "<why they should be merged>"
        }}
    ],
    "unique_entities": ["<id1>", "<id2>", ...]
}}"""


RELATION_NORMALIZATION_PROMPT = """Normalize these relation types to a canonical form.

RELATIONS USED:
{relations_json}

RULES:
1. Only merge relations that are TRUE SYNONYMS (e.g., "works_at" and "employed_by")
2. Do NOT merge relations that have different semantics even if they seem related
   - "ASSOCIATED_WITH" and "CAUSES" are DIFFERENT — keep them separate
   - "TREATS" and "AFFECTS" are DIFFERENT — keep them separate
   - "CONTRIBUTES_TO" and "ASSOCIATED_WITH" are DIFFERENT — keep them separate
   - "IS_MARKER_FOR" and "ASSOCIATED_WITH" are DIFFERENT — keep them separate
3. Preserve semantic granularity — it is better to have too many relation types than too few
4. Identify true inverse relations (e.g., "parent_of" vs "child_of")
5. Only include relations that ACTUALLY NEED normalizing in the output. If a relation is already in good form, do NOT include it.

Return:
{{
    "normalizations": [
        {{
            "original": "<original relation>",
            "normalized": "<canonical form>",
            "is_inverse": <true/false>,
            "reason": "<why these are truly synonymous>"
        }}
    ],
    "canonical_relations": ["<relation1>", "<relation2>", ...]
}}"""


class KnowledgeOrganizer(BaseAgent):
    """
    Knowledge Organizer Agent - Final KG integration and maintenance.
    
    Responsibilities:
    1. Integrate verified extractions into knowledge graph
    2. Entity deduplication and merging
    3. Relation normalization
    4. Maintain knowledge graph consistency
    5. Provide export and statistics
    
    Uses SharedMemory to:
    - Track entity aliases for deduplication
    - Store integration history
    
    Uses MessageBus to:
    - Receive approved extractions
    - Report integration results
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        governed_kg: Optional[GovernedKnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        enable_deduplication: bool = True,
        enable_normalization: bool = True,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        super().__init__(
            name="KnowledgeOrganizer",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
        )
        self.governed_kg = governed_kg
        self.enable_deduplication = enable_deduplication
        self.enable_normalization = enable_normalization
        self.retrieval_config = retrieval_config or RetrievalConfig()
        # Set to False permanently for this run if the embedding backend errors
        # or is unreachable, so a downed server degrades to lexical resolution
        # instead of crashing or hanging in retry backoff.
        self._embeddings_available = False
        if self.retrieval_config.use_vectors:
            try:
                from multi_agent_kg.llm.openai_client import _ollama_is_up

                self._embeddings_available = _ollama_is_up()
            except Exception:
                self._embeddings_available = False
        
        # Track relation normalizations
        self.relation_mappings: Dict[str, str] = {}
        
        # Integration statistics
        self.integration_stats = {
            "entities_added": 0,
            "entities_merged": 0,
            "triples_added": 0,
            "triples_updated": 0,
            "relations_normalized": 0,
            "relations_schema_mapped": 0,
            "relations_schema_rejected": 0,
            "governance_repairs": 0,
        }

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Integrate extractions into knowledge graph.
        
        Args:
            context: Processing context
            entities: Verified entities
            triples: Verified triples
            
        Returns:
            ExtractionResult with integration results
        """
        self.stats["calls"] += 1
        
        entities = entities or context.entities or []
        triples = triples or context.relations or []
        
        # Process incoming messages
        messages = self.receive_messages()
        for msg in messages:
            if msg.comm_type == CommunicationType.DELEGATE and msg.content.get("action") == "integrate":
                entities = msg.content.get("entities", entities)
                triples = msg.content.get("triples", triples)
        
        # Step 1: Entity deduplication
        print(f"\n" + "="*70)
        print(f"[ORGANIZER DEBUG] Entity Deduplication")
        print(f"="*70)
        print(f"  Input entities: {len(entities)}")
        
        if self.enable_deduplication and entities:
            entities, merged_count = self._deduplicate_entities(entities)
            self.integration_stats["entities_merged"] += merged_count
            print(f"  After deduplication: {len(entities)} (merged: {merged_count})")
        
        # Step 2: Relation normalization
        print(f"\n[ORGANIZER DEBUG] Relation Normalization")
        print(f"  Input triples: {len(triples)}")
        
        if self.enable_normalization and triples:
            triples, normalized_count = self._normalize_relations(triples)
            self.integration_stats["relations_normalized"] += normalized_count
            print(f"  After normalization: {len(triples)} (normalized: {normalized_count})")
        
        # Step 3: Integrate into knowledge graph
        print(f"\n[ORGANIZER DEBUG] KG Integration")
        print(f"  Entities to add: {len(entities)}")
        print(f"  Triples to add: {len(triples)}")
        print(f"  KG before: {len(self.knowledge_graph.entities)} entities, {len(self.knowledge_graph.triples)} triples")
        
        if self.knowledge_graph:
            added_entities, added_triples = self._integrate_to_kg(
                entities,
                triples,
                context.document_id,
            )
            self.integration_stats["entities_added"] += added_entities
            self.integration_stats["triples_added"] += added_triples
            print(f"  KG after: {len(self.knowledge_graph.entities)} entities, {len(self.knowledge_graph.triples)} triples")
            print(f"  Actually added: {added_entities} entities, {added_triples} triples")
        
        # Step 4: Update shared memory with aliases
        if self.shared_memory:
            self._update_memory(entities, triples, context.document_id)
        
        self.log(
            f"Integrated {len(entities)} entities, {len(triples)} triples "
            f"(merged: {self.integration_stats['entities_merged']}, "
            f"normalized: {self.integration_stats['relations_normalized']})"
        )
        
        return ExtractionResult(
            items={
                "integrated_entities": entities,
                "integrated_triples": triples,
            },
            confidence=1.0,
            metadata={
                "document_id": context.document_id,
                "integration_stats": self.integration_stats.copy(),
                "kg_stats": self.get_kg_stats(),
            },
        )

    def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> tuple:
        """Deduplicate entities using LLM."""
        if len(entities) < 2:
            return entities, 0
        
        # First check for obvious duplicates (same name, different case)
        obvious_merges, remaining = self._find_obvious_duplicates(entities)
        
        # Use deterministic semantic matching before spending an LLM call.
        semantic_merges, remaining = self._find_semantic_duplicates(remaining)
        obvious_merges.extend(semantic_merges)
        if self.shared_memory:
            for group in semantic_merges:
                canonical_id = group.get("canonical_id")
                for alias_id in group.get("merge_ids", []):
                    self.shared_memory.register_entity_alias(alias_id, canonical_id)

        # Use LLM for non-obvious cases
        if len(remaining) > 1:
            entities_json = json.dumps([
                {
                    "id": e.get("id", e.get("text", "")),
                    "text": e.get("text", ""),
                    "type": e.get("type", ""),
                }
                for e in remaining
            ], indent=2)
            
            prompt = ENTITY_DEDUP_PROMPT.format(entities_json=entities_json)
            
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at entity resolution. Identify duplicates carefully.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            
            # Handle both dict and list responses from LLM
            if isinstance(result, dict):
                merge_groups = result.get("merge_groups", [])
            elif isinstance(result, list):
                merge_groups = result  # LLM returned list directly
            else:
                merge_groups = []
            
            # Apply merges
            for group in merge_groups:
                canonical_id = group.get("canonical_id")
                canonical_name = group.get("canonical_name")
                merge_ids = group.get("merge_ids", [])
                
                if canonical_id and merge_ids:
                    # Register aliases in shared memory
                    if self.shared_memory:
                        for alias_id in merge_ids:
                            self.shared_memory.register_entity_alias(alias_id, canonical_id)
                    
                    # Remove merged entities
                    remaining = [e for e in remaining if e.get("id", e.get("text", "")) not in merge_ids]
                    obvious_merges.extend(merge_groups)
        
        merged_count = len(entities) - len(remaining)
        return remaining, merged_count

    def _find_semantic_duplicates(
        self,
        entities: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Collapse near-duplicate entities with type-aware lexical similarity."""
        if len(entities) < 2:
            return [], entities

        merges = []
        consumed: Set[int] = set()
        remaining: List[Dict[str, Any]] = []

        def normalized_text(entity: Dict[str, Any]) -> str:
            text = entity.get("text") or entity.get("id", "")
            return " ".join(text.lower().replace("_", " ").split())

        def trigram_vector(text: str) -> Dict[str, int]:
            padded = f"  {text}  "
            vector: Dict[str, int] = {}
            for idx in range(max(len(padded) - 2, 1)):
                gram = padded[idx:idx + 3]
                vector[gram] = vector.get(gram, 0) + 1
            return vector

        def cosine_similarity(left: Dict[str, int], right: Dict[str, int]) -> float:
            shared = set(left) & set(right)
            numerator = sum(left[token] * right[token] for token in shared)
            left_norm = sqrt(sum(value * value for value in left.values()))
            right_norm = sqrt(sum(value * value for value in right.values()))
            if not left_norm or not right_norm:
                return 0.0
            return numerator / (left_norm * right_norm)

        normalized = [normalized_text(entity) for entity in entities]
        vectors = [trigram_vector(text) for text in normalized]
        types = [(entity.get("type") or "").upper() for entity in entities]

        # Optional embedding-based similarity: catches synonym/abbreviation
        # duplicates ("NYC" vs "New York City") that trigram overlap misses.
        embedding_matrix = None
        if self._embeddings_available and len(entities) >= 2:
            try:
                import numpy as np

                from multi_agent_kg.llm.openai_client import embed_passages

                raw = np.asarray(
                    embed_passages(normalized, model=self.retrieval_config.embedding_model),
                    dtype=np.float32,
                )
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embedding_matrix = raw / norms
            except Exception as exc:
                print(f"  WARNING: embedding dedup disabled ({exc})")
                self._embeddings_available = False

        def embedding_similarity(left: int, right: int) -> float:
            if embedding_matrix is None:
                return 0.0
            return float(embedding_matrix[left] @ embedding_matrix[right])

        for idx, entity in enumerate(entities):
            if idx in consumed:
                continue
            canonical_id = entity.get("id", entity.get("text", ""))
            merged_ids: List[str] = []
            merge_reason = "type-aware trigram similarity"
            for other_idx in range(idx + 1, len(entities)):
                if other_idx in consumed:
                    continue
                if types[idx] and types[other_idx] and types[idx] != types[other_idx]:
                    continue
                similarity = cosine_similarity(vectors[idx], vectors[other_idx])
                if similarity < 0.88:
                    if embedding_similarity(idx, other_idx) < 0.85:
                        continue
                    merge_reason = "embedding similarity"
                other_id = entities[other_idx].get("id", entities[other_idx].get("text", ""))
                merged_ids.append(other_id)
                consumed.add(other_idx)
            if merged_ids:
                merges.append(
                    {
                        "canonical_id": canonical_id,
                        "canonical_name": entity.get("text", canonical_id),
                        "merge_ids": merged_ids,
                        "reason": merge_reason,
                    }
                )
            remaining.append(entity)

        return merges, remaining

    def _find_obvious_duplicates(
        self,
        entities: List[Dict[str, Any]],
    ) -> tuple:
        """Find obvious duplicates (same name, case insensitive)."""
        seen: Dict[str, Dict] = {}
        remaining = []
        merges = []
        
        for entity in entities:
            text = entity.get("text", "").lower().strip()
            if text in seen:
                # Merge
                merges.append({
                    "canonical": seen[text].get("id", seen[text].get("text")),
                    "merged": entity.get("id", entity.get("text")),
                })
            else:
                seen[text] = entity
                remaining.append(entity)
        
        return merges, remaining

    def _normalize_relations(
        self,
        triples: List[Dict[str, Any]],
    ) -> tuple:
        """Normalize relation types."""
        if not triples:
            return triples, 0

        # Drop any non-dict item that slipped through upstream LLM parsing.
        # Without this, a malformed row (e.g. a nested list) crashes the whole
        # document with "'list' object has no attribute 'get'".
        bad = sum(1 for t in triples if not isinstance(t, dict))
        if bad:
            print(f"  WARNING: dropping {bad} non-dict triple(s) before relation normalization")
            triples = [t for t in triples if isinstance(t, dict)]
            if not triples:
                return triples, 0

        allowed_relations = self._allowed_relation_types()
        if allowed_relations:
            normalized_count = 0
            for triple in triples:
                original_rel = triple.get("relation", "")
                normalized_rel, allowed = self._enforce_relation_schema(
                    original_rel,
                    allowed_relations,
                )
                if allowed and normalized_rel != original_rel:
                    triple["original_relation"] = original_rel
                    triple["relation"] = normalized_rel
                    normalized_count += 1
            return triples, normalized_count
        
        # Collect unique relations
        relations = list(set(t.get("relation", "") for t in triples if t.get("relation")))
        
        if len(relations) < 2:
            return triples, 0

        # Embedding-similarity pre-clustering: map near-duplicate relation
        # surface forms to a canonical (most frequent) form before the LLM call.
        # This curbs relation-type drift (WORKS_AT / EMPLOYED_BY / IS_EMPLOYED_AT).
        embedding_count = self._cluster_relations_by_embedding(relations, triples)

        # Check cache first
        uncached_relations = [r for r in relations if r not in self.relation_mappings]
        
        if uncached_relations:
            relations_json = json.dumps(uncached_relations, indent=2)
            
            prompt = RELATION_NORMALIZATION_PROMPT.format(relations_json=relations_json)
            
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at relation normalization. Be consistent.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            
            # Update cache
            for norm in result.get("normalizations", []):
                original = norm.get("original", "")
                normalized = norm.get("normalized", original)
                self.relation_mappings[original] = normalized
        
        # Apply normalizations
        normalized_count = 0
        for triple in triples:
            original_rel = triple.get("relation", "")
            if original_rel in self.relation_mappings:
                normalized_rel = self.relation_mappings[original_rel]
                if normalized_rel != original_rel:
                    triple["original_relation"] = original_rel
                    triple["relation"] = normalized_rel
                    normalized_count += 1
        
        return triples, normalized_count + embedding_count

    def _cluster_relations_by_embedding(
        self,
        relations: List[str],
        triples: List[Dict[str, Any]],
        threshold: float = 0.88,
    ) -> int:
        """Cluster near-duplicate relations via embedding cosine similarity.

        For each cluster of similar relations, the most frequent surface form
        becomes canonical; all others are added to self.relation_mappings and
        rewritten on the triples in place. Returns the count of triples rewritten.

        Falls back to a no-op when embeddings are unavailable.
        """
        if not self._embeddings_available or len(relations) < 2:
            return 0
        try:
            import numpy as np
            from multi_agent_kg.llm.openai_client import embed_passages
        except Exception:
            return 0

        # Humanize relation names for better embedding semantics
        texts = [r.replace("_", " ").lower() for r in relations]
        try:
            vectors = np.asarray(embed_passages(texts), dtype=np.float32)
        except Exception as exc:
            print(f"  [RelationCluster] embedding unavailable ({exc}); skipping")
            self._embeddings_available = False
            return 0
        if vectors.ndim != 2 or vectors.shape[0] != len(relations):
            return 0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        sims = unit @ unit.T

        # Frequency of each relation across triples (for canonical selection)
        freq: Dict[str, int] = {}
        for t in triples:
            rel = t.get("relation", "")
            if rel:
                freq[rel] = freq.get(rel, 0) + 1

        assigned: Dict[int, int] = {}  # row -> cluster canonical row
        for i in range(len(relations)):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(i + 1, len(relations)):
                if j in assigned:
                    continue
                if float(sims[i, j]) >= threshold:
                    cluster.append(j)
            # Canonical = most frequent (tie-break: longest descriptive name)
            canonical_row = max(
                cluster,
                key=lambda r: (freq.get(relations[r], 0), len(relations[r])),
            )
            for r in cluster:
                assigned[r] = canonical_row

        rewritten = 0
        cluster_map: Dict[str, str] = {}
        for row, canon_row in assigned.items():
            if row == canon_row:
                continue
            original = relations[row]
            canonical = relations[canon_row]
            if original != canonical:
                cluster_map[original] = canonical
                self.relation_mappings[original] = canonical

        if cluster_map:
            for triple in triples:
                rel = triple.get("relation", "")
                if rel in cluster_map:
                    triple["original_relation"] = rel
                    triple["relation"] = cluster_map[rel]
                    rewritten += 1
            print(f"  [RelationCluster] merged {len(cluster_map)} relation variant(s) via embedding similarity")
        return rewritten

    def _integrate_to_kg(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> tuple:
        """Integrate into knowledge graph with entity-name resolution.
        
        Builds a lookup from entity text/labels → entity ID so that
        triple subjects/objects (which use text names) get resolved to
        real entity IDs instead of creating phantom entities.
        """
        added_entities = 0
        added_triples = 0
        skipped_triples = 0

        # ── Filter garbage entities ──────────────────────────────────
        import re
        clean_entities = []
        _GARBAGE_PHRASES = {
            "our findings", "this study", "we", "they", "it", "its",
            "the study", "results", "data", "analysis", "the method",
            "the procedure", "the results", "the analysis", "the model",
            "the approach", "the system", "the technique", "the treatment",
            "the patient", "the patients", "the group", "the sample",
            "the outcome", "the effect", "the association", "the relationship",
            "these findings", "these results", "our study", "our results",
            "the present study", "the current study", "previous studies",
            "a study", "a method", "an approach", "the authors",
            "placebo", "control group", "baseline", "follow-up",
            # Pronoun / reference phrases that coreference failed to resolve
            "our approach", "this method", "this approach", "this system",
            "this information", "this technique", "this model",
            "these methods", "these models",
            "the proposed method", "the proposed approach",
        }
        for entity in entities:
            eid = entity.get("id", entity.get("text", ""))
            etext = entity.get("text", eid)
            # Skip empty, pure-number, or trivially short/generic entities
            if not etext or not etext.strip():
                continue
            stripped = etext.strip()
            if re.fullmatch(r'\d+', stripped):
                # Numeric-only ID *and* numeric-only text → garbage
                if re.fullmatch(r'\d+', eid):
                    continue
            if stripped.lower() in _GARBAGE_PHRASES:
                continue
            if len(stripped) < 2:
                continue
            # Skip entities with garbage types
            etype = entity.get("type", "").upper()
            if etype in {
                "STATISTICAL_METHOD", "STUDY_DESIGN", "ANALYSIS_TECHNIQUE",
                "STATISTICAL_MODEL", "UNRESOLVED",
            }:
                continue
            clean_entities.append(entity)
        # ── Consolidate entity types ─────────────────────────────────
        _TYPE_CONSOLIDATION = {
            "TASK": "Task",
            "METHOD": "Method",
            "METRIC": "Metric",
            "MATERIAL": "Material",
            "OTHERSCIENTIFICTERM": "OtherScientificTerm",
            "CLINICAL_BIOMARKER": "BIOLOGICAL_MARKER",
            "BIOMARKER": "BIOLOGICAL_MARKER",
            "IMMUNE_PROCESS": "BIOLOGICAL_PROCESS",
            "PATHOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
            "COMPLEMENT_PATHWAY": "BIOLOGICAL_PROCESS",
            "BIOLOGICAL_PATHWAY": "BIOLOGICAL_PROCESS",
            "PHYSIOLOGICAL_MEASUREMENT": "CARDIOVASCULAR_MEASUREMENT",
            "IMAGING_MEASUREMENT": "CARDIOVASCULAR_MEASUREMENT",
            "PHYSIOLOGICAL_PARAMETER": "CARDIOVASCULAR_MEASUREMENT",
            "METABOLIC_MEASUREMENT": "CARDIOVASCULAR_MEASUREMENT",
            "VASCULAR_CONDITION": "METABOLIC_CONDITION",
            "PATHOLOGICAL_CONDITION": "METABOLIC_CONDITION",
            "CARDIOVASCULAR_DISEASE": "METABOLIC_CONDITION",
            "RENAL_CONDITION": "METABOLIC_CONDITION",
            "BIOLOGICAL_TISSUE": "ANATOMICAL_STRUCTURE",
            "ORGAN": "ANATOMICAL_STRUCTURE",
            "DRUG": "MEDICATION",
            "THERAPEUTIC_AGENT": "MEDICATION",
        }
        for entity in clean_entities:
            etype = entity.get("type", "")
            if etype in _TYPE_CONSOLIDATION:
                entity["original_type"] = etype
                entity["type"] = _TYPE_CONSOLIDATION[etype]

        print(f"  Filtered entities: {len(entities)} → {len(clean_entities)} "
              f"(removed {len(entities) - len(clean_entities)} garbage)")
        entities = clean_entities

        # ── Build name → entity_id lookup ────────────────────────────
        # This is the critical mapping that prevents phantom entities.
        # The entity extractor assigns canonical IDs (sometimes numeric),
        # but the relation extractor references entities by text name.
        name_to_id: Dict[str, str] = {}
        for entity in entities:
            eid = entity.get("id", entity.get("text", ""))
            etext = entity.get("text", eid)
            # Map the text form → canonical ID
            name_to_id[etext.lower().strip()] = eid
            # Map the ID itself
            name_to_id[eid.lower().strip()] = eid
            # Map any aliases/mentions
            for mention in entity.get("mentions", []):
                name_to_id[mention.lower().strip()] = eid

        # Also map existing KG entities
        for existing_id, existing_entity in self.knowledge_graph.entities.items():
            name_to_id[existing_id.lower().strip()] = existing_id
            for label in existing_entity.labels:
                name_to_id[label.lower().strip()] = existing_id

        # Also check shared memory for aliases
        if self.shared_memory and hasattr(self.shared_memory, 'entity_aliases'):
            for alias, canonical in self.shared_memory.entity_aliases.items():
                name_to_id[alias.lower().strip()] = canonical
                
        resolve_misses: Dict[str, int] = {}
        embedding_resolutions: Dict[str, str] = {}

        # Lazily-built embedding index over all known entity names. Tier 3 of
        # resolution: catches surface-form mismatches ("CEO of Apple",
        # "Apple's retail outlets") that exact/normalized matching misses,
        # which previously spawned UNRESOLVED duplicate nodes.
        _name_index: Dict[str, Any] = {"index": None}

        def _get_name_index():
            if not self._embeddings_available:
                return None
            if _name_index["index"] is None:
                try:
                    from multi_agent_kg.core.vector_index import VectorIndex

                    index = VectorIndex(model=self.retrieval_config.embedding_model)
                    index.upsert({known_name: known_name for known_name in name_to_id})
                    _name_index["index"] = index
                except Exception as exc:
                    print(f"  WARNING: embedding resolution disabled ({exc})")
                    self._embeddings_available = False
                    return None
            return _name_index["index"]

        def _resolve_entity_name(name: Optional[str]) -> Optional[str]:
            """Resolve a triple subject/object text to an entity ID."""
            if not name:
                return None
            key = str(name).lower().strip()
            if key in name_to_id:
                return name_to_id[key]
            normalized = " ".join(key.replace("_", " ").replace("-", " ").split())
            for known_name, known_id in name_to_id.items():
                known_normalized = " ".join(
                    known_name.replace("_", " ").replace("-", " ").split()
                )
                if known_normalized == normalized:
                    return known_id
            index = _get_name_index()
            if index is not None:
                try:
                    hits = index.search(
                        normalized,
                        top_k=1,
                        min_score=self.retrieval_config.resolution_min_score,
                    )
                except Exception as exc:
                    print(f"  WARNING: embedding resolution disabled ({exc})")
                    self._embeddings_available = False
                    hits = []
                if hits:
                    matched_id = name_to_id[hits[0][0]]
                    embedding_resolutions[str(name)] = matched_id
                    return matched_id
            miss_key = str(name)
            resolve_misses[miss_key] = resolve_misses.get(miss_key, 0) + 1
            return None

        allowed_relations = self._allowed_relation_types()
        seen_triples: Set[tuple] = set()
        skipped_triple_reasons = {
            "missing_field": 0,
            "bad_relation": 0,
            "schema_rejected": 0,
            "self_reference": 0,
            "duplicate": 0,
            "unresolved_entity": 0,
            "add_failed": 0,
        }

        # ── Add entities ─────────────────────────────────────────────
        for entity in entities:
            entity_id = entity.get("id", entity.get("text", ""))
            etext = entity.get("text", entity_id)
            # Use text as ID if the ID is purely numeric (fixes numeric IDs)
            if re.fullmatch(r'\d+', entity_id) and etext and not re.fullmatch(r'\d+', etext):
                old_id = entity_id
                entity_id = etext.lower().replace(" ", "_")
                # Update the lookup
                name_to_id[etext.lower().strip()] = entity_id
                name_to_id[old_id] = entity_id
                for mention in entity.get("mentions", []):
                    name_to_id[mention.lower().strip()] = entity_id
            else:
                # Clean up _1, _group suffixes from entity IDs
                clean_id = re.sub(r'(?<=\w{3})_\d+$', '', entity_id)
                clean_id = re.sub(r'_group$', '', clean_id)
                if clean_id != entity_id and clean_id not in self.knowledge_graph.entities:
                    name_to_id[entity_id.lower().strip()] = clean_id
                    entity_id = clean_id

            if entity_id not in self.knowledge_graph.entities:
                # Build labels from all available sources: text, mentions, labels
                entity_labels = []
                if etext:
                    entity_labels.append(etext)
                for lbl in entity.get("labels", []):
                    if lbl and lbl not in entity_labels:
                        entity_labels.append(lbl)
                for mention in entity.get("mentions", []):
                    if mention and mention not in entity_labels:
                        entity_labels.append(mention)
                if not entity_labels:
                    entity_labels = [entity_id]

                if self.governed_kg:
                    self.governed_kg.add_entity(
                        entity_id=entity_id,
                        labels=entity_labels,
                        entity_type=entity.get("type", "UNKNOWN"),
                        metadata={
                            "source_document": document_id,
                            "confidence": entity.get("confidence", 0.7),
                            "source_segment": entity.get("source_segment"),
                            "source_segments": entity.get("source_segments", []),
                            "source_text": entity.get("source_text", ""),
                            "source_texts": entity.get("source_texts", []),
                            "extraction_method": entity.get("extraction_method", ""),
                        },
                    )
                    if entity.get("candidate_domains"):
                        self.governed_kg.assign_entity_to_domains(
                            entity_id,
                            entity.get("candidate_domains", []),
                        )
                else:
                    self.knowledge_graph.add_entity(
                        entity_id=entity_id,
                        labels=entity_labels,
                        entity_type=entity.get("type", "UNKNOWN"),
                        metadata={
                            "source_document": document_id,
                            "confidence": entity.get("confidence", 0.7),
                            "source_segment": entity.get("source_segment"),
                            "source_segments": entity.get("source_segments", []),
                            "source_text": entity.get("source_text", ""),
                            "source_texts": entity.get("source_texts", []),
                            "extraction_method": entity.get("extraction_method", ""),
                        },
                    )
                added_entities += 1

        # ── Add triples (with entity resolution) ─────────────────────
        # Filter out meaningless relation types
        _BAD_RELATIONS = {
            "DRUG_EXAMPLE", "EXAMPLE_OF",
        }
        for triple in triples:
            raw_subj = triple.get("subject", "")
            raw_obj = triple.get("object", "")
            relation = triple.get("relation", "")

            if not raw_subj or not raw_obj or not relation:
                skipped_triple_reasons["missing_field"] += 1
                skipped_triples += 1
                continue
            if relation.upper() in _BAD_RELATIONS:
                skipped_triple_reasons["bad_relation"] += 1
                skipped_triples += 1
                continue
            relation, relation_allowed = self._enforce_relation_schema(
                relation,
                allowed_relations,
            )
            if not relation_allowed:
                self.integration_stats["relations_schema_rejected"] += 1
                skipped_triple_reasons["schema_rejected"] += 1
                skipped_triples += 1
                continue

            # Resolve subject and object to known entity IDs
            resolved_subj = _resolve_entity_name(raw_subj) or _resolve_entity_name(
                triple.get("subject_id", "")
            )
            resolved_obj = _resolve_entity_name(raw_obj) or _resolve_entity_name(
                triple.get("object_id", "")
            )

            if not resolved_subj or not resolved_obj:
                skipped_triple_reasons["unresolved_entity"] += 1
                skipped_triples += 1
                continue

            triple_key = (resolved_subj, relation, resolved_obj)
            if resolved_subj == resolved_obj:
                skipped_triple_reasons["self_reference"] += 1
                skipped_triples += 1
                continue
            if triple_key in seen_triples:
                skipped_triple_reasons["duplicate"] += 1
                skipped_triples += 1
                continue
            seen_triples.add(triple_key)

            if self.governed_kg:
                triple_metadata = coerce_metadata(triple.get("metadata", {}))
                already_repaired = bool(
                    triple.get("governance_repair")
                    or triple_metadata.get("governance_repair")
                )
                decision = self.governed_kg.propose_triple(
                    subject=resolved_subj,
                    relation=relation,
                    obj=resolved_obj,
                    confidence=triple.get("final_confidence", triple.get("confidence", 0.7)),
                    source=document_id,
                    metadata={
                        "evidence": (triple.get("supporting_evidence") or triple.get("evidence") or ""),
                        "verification_status": triple.get("verification_status", "unknown"),
                        "original_subject": raw_subj,
                        "original_object": raw_obj,
                    },
                )
                if (
                    decision.action in {"reject", "escalate"}
                    and self.governed_kg.governance_mode in {"strict", "triage"}
                    and not already_repaired
                ):
                    # Single-attempt repair semantics: once a triple has been
                    # repaired and re-proposed, we accept the second decision
                    # as final rather than entering another repair cycle.
                    repaired = self._repair_triple_for_governance(
                        subject=resolved_subj,
                        relation=relation,
                        obj=resolved_obj,
                        evidence=(triple.get("supporting_evidence") or triple.get("evidence") or ""),
                        allowed_relations=allowed_relations,
                        rationale=decision.rationale,
                    )
                    if repaired is not None:
                        self.integration_stats["governance_repairs"] += 1
                        decision = self.governed_kg.propose_triple(
                            subject=repaired["subject"],
                            relation=repaired["relation"],
                            obj=repaired["object"],
                            confidence=triple.get("final_confidence", triple.get("confidence", 0.7)),
                            source=document_id,
                            metadata={
                                "evidence": (triple.get("supporting_evidence") or triple.get("evidence") or ""),
                                "verification_status": triple.get("verification_status", "unknown"),
                                "original_subject": raw_subj,
                                "original_object": raw_obj,
                                "governance_repair": True,
                            },
                        )
                result = decision if decision.committed else None
            else:
                result = self.knowledge_graph.add_triple(
                    subject=resolved_subj,
                    relation=relation,
                    obj=resolved_obj,
                    confidence=triple.get("final_confidence", triple.get("confidence", 0.7)),
                    source=document_id,
                    metadata={
                        "evidence": (triple.get("supporting_evidence") or triple.get("evidence") or ""),
                        "verification_status": triple.get("verification_status", "unknown"),
                        "original_subject": raw_subj,
                        "original_object": raw_obj,
                    },
                )
            if result is not None:
                added_triples += 1
            else:
                skipped_triple_reasons["add_failed"] += 1
                skipped_triples += 1  # Duplicate

        print(f"  Entity resolution: mapped {len(name_to_id)} name variants")
        if embedding_resolutions:
            print(f"  Embedding-tier resolutions: {len(embedding_resolutions)}")
            for raw_name, matched_id in list(embedding_resolutions.items())[:10]:
                print(f"    - '{raw_name}' -> {matched_id}")
        miss_count = sum(resolve_misses.values())
        print(f"  Entity resolution misses: {miss_count}")
        if resolve_misses:
            for miss_name, count in sorted(resolve_misses.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - Miss: '{miss_name}' ({count} times)")
        
        print(f"  Triples skipped (dup/invalid): {skipped_triples}")
        if skipped_triples:
            print(f"  Triple skip reasons: {skipped_triple_reasons}")
            
        self.integration_stats["kg_triples_added"] = added_triples
        self.integration_stats["kg_entities_added"] = added_entities
        self.integration_stats["skipped_triple_reasons"] = skipped_triple_reasons
        self.integration_stats["resolve_entity_name_misses"] = miss_count
        self.integration_stats["resolve_misses"] = dict(resolve_misses)
        self.integration_stats["embedding_resolutions"] = dict(embedding_resolutions)
        self.integration_stats["relations_schema_rejected"] = self.integration_stats.get("relations_schema_rejected", 0)
        self.integration_stats["triple_skip_reasons"] = skipped_triple_reasons
        return added_entities, added_triples

    def _allowed_relation_types(self) -> Set[str]:
        if not self.governed_kg or not self.governed_kg.org_chart.domains:
            return set()
        allowed: Set[str] = set()
        for domain in self.governed_kg.org_chart.domains:
            allowed.update(domain.relation_schema.keys())
            domain_metadata = coerce_metadata(domain.metadata)
            allowed.update(domain_metadata.get("seed_relation_types", []))
        return {relation for relation in allowed if relation}

    def _enforce_relation_schema(
        self,
        relation: str,
        allowed_relations: Set[str],
    ) -> Tuple[str, bool]:
        if not allowed_relations:
            return relation, True
        if relation in allowed_relations:
            return relation, True

        canonical_relation = relation.upper().replace("-", "_").replace(" ", "_")
        canonical_allowed = {
            allowed: allowed.upper().replace("-", "_").replace(" ", "_")
            for allowed in allowed_relations
        }
        for allowed, normalized in canonical_allowed.items():
            if normalized == canonical_relation:
                self.integration_stats["relations_schema_mapped"] += 1
                return allowed, True

        # Permissive / audit_only modes admit open-world relations; governance
        # records them via the audit log and memory cards rather than rejecting
        # at integration. Strict / triage modes still gate at the schema.
        permissive_modes = {"permissive", "audit_only"}
        if self.governed_kg and self.governed_kg.governance_mode in permissive_modes:
            self.integration_stats["relations_schema_admitted_open_world"] = (
                self.integration_stats.get("relations_schema_admitted_open_world", 0) + 1
            )
            return relation, True
        return relation, False

    def _repair_triple_for_governance(
        self,
        *,
        subject: str,
        relation: str,
        obj: str,
        evidence: str,
        allowed_relations: Set[str],
        rationale: str,
    ) -> Optional[Dict[str, str]]:
        if not allowed_relations:
            return None
        prompt = f"""A triple was rejected or escalated during governed KG creation.

Current triple:
({subject}) -[{relation}]-> ({obj})

Allowed relation types:
{sorted(allowed_relations)}

Evidence:
{evidence[:1200]}

Governance rationale:
{rationale}

If the triple can be repaired to match the allowed relation schema without changing the meaning,
return JSON:
{{
  "action": "revise",
  "subject": "{subject}",
  "relation": "<allowed relation>",
  "object": "{obj}"
}}

Otherwise return:
{{"action": "reject"}}

Return ONLY JSON."""
        try:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You repair candidate triples conservatively. Return only valid JSON.",
                tier=ModelTier.MEDIUM,
                max_tokens=512,
            )
        except Exception as e:
            print(f"  [GOVERNANCE] _repair_triple_for_governance failed: {e}")
            return None

        if result.get("action") != "revise":
            return None
        revised_relation, relation_allowed = self._enforce_relation_schema(
            result.get("relation", relation),
            allowed_relations,
        )
        if not relation_allowed:
            return None
        return {
            "subject": result.get("subject", subject),
            "relation": revised_relation,
            "object": result.get("object", obj),
        }

    def _update_memory(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Update shared memory with integration results."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "integrated_entities": [e.get("id", e.get("text")) for e in entities],
                "integrated_triples": len(triples),
                "document_id": document_id,
            },
        )

    def get_kg_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.knowledge_graph:
            return {}
        
        # Count relation types
        relation_counts = defaultdict(int)
        for triple in self.knowledge_graph.triples:
            relation_counts[triple.relation] += 1
        
        # Count entity types
        entity_type_counts = defaultdict(int)
        for entity in self.knowledge_graph.entities.values():
            entity_type_counts[entity.type] += 1
        
        return {
            "total_entities": len(self.knowledge_graph.entities),
            "total_triples": len(self.knowledge_graph.triples),
            "entity_types": dict(entity_type_counts),
            "relation_types": dict(relation_counts),
            "unique_relations": len(relation_counts),
        }

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the complete knowledge graph."""
        if not self.knowledge_graph:
            return {"entities": [], "triples": []}
        
        # Build entity ID -> readable text lookup
        entity_text_map = {}
        entities = []
        for entity_id, entity in self.knowledge_graph.entities.items():
            # Use first label if available, otherwise convert ID to readable text
            text = entity.labels[0] if entity.labels else entity_id.replace("_", " ")
            entity_text_map[entity_id] = text
            entities.append({
                "id": entity_id,
                "text": text,
                "labels": entity.labels,
                "type": entity.type,
                "metadata": entity.metadata,
            })

        triples = []
        for i, triple in enumerate(self.knowledge_graph.triples):
            # Use readable text for subject/object so evaluators can match
            subj_text = entity_text_map.get(triple.subject, triple.subject.replace("_", " "))
            obj_text = entity_text_map.get(triple.object, triple.object.replace("_", " "))
            triples.append({
                "id": f"triple_{i}",
                "subject": subj_text,
                "relation": triple.relation,
                "object": obj_text,
                "subject_id": triple.subject,
                "object_id": triple.object,
                "confidence": triple.confidence,
                "source": triple.source,
            })
        
        return {
            "entities": entities,
            "triples": triples,
            "stats": self.get_kg_stats(),
            "integration_stats": self.integration_stats,
        }
