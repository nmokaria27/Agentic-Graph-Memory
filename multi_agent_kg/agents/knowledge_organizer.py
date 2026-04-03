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

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity, Triple
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig


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

Look for:
1. Synonymous relations (e.g., "works_at" vs "employed_by")
2. Inverse relations (e.g., "parent_of" vs "child_of")
3. Relations that should be standardized

Return:
{{
    "normalizations": [
        {{
            "original": "<original relation>",
            "normalized": "<canonical form>",
            "is_inverse": <true/false>,
            "reason": "<normalization reason>"
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
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        enable_deduplication: bool = True,
        enable_normalization: bool = True,
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
        self.enable_deduplication = enable_deduplication
        self.enable_normalization = enable_normalization
        
        # Track relation normalizations
        self.relation_mappings: Dict[str, str] = {}
        
        # Integration statistics
        self.integration_stats = {
            "entities_added": 0,
            "entities_merged": 0,
            "triples_added": 0,
            "triples_updated": 0,
            "relations_normalized": 0,
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
        
        # Step 2.5: Triple deduplication (alias-aware)
        print(f"\n[ORGANIZER DEBUG] Triple Deduplication")
        print(f"  Input triples: {len(triples)}")

        if triples:
            triples, dedup_count = self._deduplicate_triples(triples, entities)
            print(f"  After deduplication: {len(triples)} (removed: {dedup_count})")

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
            
            merge_groups = result.get("merge_groups", [])
            
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
        
        # Collect unique relations
        relations = list(set(t.get("relation", "") for t in triples if t.get("relation")))
        
        if len(relations) < 2:
            return triples, 0
        
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
        
        return triples, normalized_count

    def _deduplicate_triples(
        self,
        triples: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate triples and normalize entity references to canonical names.

        Steps:
        1. Build alias → canonical mapping from the entities list.
        2. Normalize subject and object fields to canonical names.
        3. Remove exact duplicates (case-insensitive on subject, relation, object).

        Returns:
            (deduplicated_triples, removed_count)
        """
        # ── Build alias → canonical mapping ──────────────────────────
        alias_to_canonical: Dict[str, str] = {}
        for entity in entities:
            canonical = entity.get("text", "").strip()
            if not canonical:
                continue
            canonical_lower = canonical.lower()
            # The canonical name maps to itself
            alias_to_canonical[canonical_lower] = canonical
            # Map every alias to the canonical form
            for alias in entity.get("aliases", []):
                alias_lower = alias.strip().lower()
                if alias_lower:
                    alias_to_canonical[alias_lower] = canonical
            # Also map mentions (some pipelines store aliases there)
            for mention in entity.get("mentions", []):
                mention_lower = mention.strip().lower()
                if mention_lower:
                    alias_to_canonical[mention_lower] = canonical

        # Also pull aliases from shared memory if available
        if self.shared_memory and hasattr(self.shared_memory, "entity_aliases"):
            for alias, canonical_id in self.shared_memory.entity_aliases.items():
                alias_lower = alias.strip().lower()
                # Resolve canonical_id to a text name if possible
                canonical_text = None
                for entity in entities:
                    eid = entity.get("id", entity.get("text", ""))
                    if eid == canonical_id:
                        canonical_text = entity.get("text", eid).strip()
                        break
                if canonical_text:
                    alias_to_canonical[alias_lower] = canonical_text

        def _canonicalize(name: str) -> str:
            """Return the canonical form of an entity name, or the original
            (stripped) name if no alias mapping exists."""
            key = name.strip().lower()
            return alias_to_canonical.get(key, name.strip())

        # ── Normalize subjects/objects and deduplicate ───────────────
        seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        deduplicated: List[Dict[str, Any]] = []
        removed = 0

        for triple in triples:
            subj = triple.get("subject", "")
            rel = triple.get("relation", "")
            obj = triple.get("object", "")

            # Normalize to canonical names
            canon_subj = _canonicalize(subj)
            canon_obj = _canonicalize(obj)

            # Write the canonical names back into the triple
            if canon_subj != subj:
                triple["original_subject"] = subj
                triple["subject"] = canon_subj
            if canon_obj != obj:
                triple["original_object"] = obj
                triple["object"] = canon_obj

            # Dedup key is case-insensitive
            key = (canon_subj.lower(), rel.lower(), canon_obj.lower())

            if key in seen:
                # Keep the triple with higher confidence
                existing = seen[key]
                existing_conf = existing.get("final_confidence", existing.get("confidence", 0.0))
                new_conf = triple.get("final_confidence", triple.get("confidence", 0.0))
                if new_conf > existing_conf:
                    # Replace with higher-confidence version
                    deduplicated = [t if t is not existing else triple for t in deduplicated]
                    seen[key] = triple
                removed += 1
            else:
                seen[key] = triple
                deduplicated.append(triple)

        return deduplicated, removed

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
            if stripped.lower() in {
                "our findings", "this study", "we", "they", "it",
                "the study", "results", "data", "analysis",
            }:
                continue
            if len(stripped) < 2:
                continue
            clean_entities.append(entity)

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

        def _resolve_entity_name(name: str) -> Optional[str]:
            """Resolve a triple subject/object text to an entity ID."""
            key = name.lower().strip()
            if key in name_to_id:
                return name_to_id[key]
            # Try fuzzy: check if name is a substring of any known entity
            for known_name, known_id in name_to_id.items():
                if len(key) > 3 and (key in known_name or known_name in key):
                    return known_id
            return None

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

            if entity_id not in self.knowledge_graph.entities:
                self.knowledge_graph.add_entity(
                    entity_id=entity_id,
                    labels=[etext] if etext != entity_id else [entity_id],
                    entity_type=entity.get("type", "UNKNOWN"),
                    metadata={
                        "source_document": document_id,
                        "confidence": entity.get("confidence", 0.7),
                    },
                )
                added_entities += 1

        # ── Add triples (with entity resolution) ─────────────────────
        for triple in triples:
            raw_subj = triple.get("subject", "")
            raw_obj = triple.get("object", "")
            relation = triple.get("relation", "")

            if not raw_subj or not raw_obj or not relation:
                skipped_triples += 1
                continue

            # Resolve subject and object to known entity IDs
            resolved_subj = _resolve_entity_name(raw_subj) or _resolve_entity_name(
                triple.get("subject_id", "")
            )
            resolved_obj = _resolve_entity_name(raw_obj) or _resolve_entity_name(
                triple.get("object_id", "")
            )

            # If we can't resolve, use the text form as a new entity
            # (but create it properly with type info, not as a phantom)
            if not resolved_subj:
                resolved_subj = raw_subj.lower().replace(" ", "_")
                if resolved_subj not in self.knowledge_graph.entities:
                    self.knowledge_graph.add_entity(
                        entity_id=resolved_subj,
                        labels=[raw_subj],
                        entity_type="UNRESOLVED",
                        metadata={"source_document": document_id, "auto_created": True},
                    )
                    name_to_id[raw_subj.lower().strip()] = resolved_subj

            if not resolved_obj:
                resolved_obj = raw_obj.lower().replace(" ", "_")
                if resolved_obj not in self.knowledge_graph.entities:
                    self.knowledge_graph.add_entity(
                        entity_id=resolved_obj,
                        labels=[raw_obj],
                        entity_type="UNRESOLVED",
                        metadata={"source_document": document_id, "auto_created": True},
                    )
                    name_to_id[raw_obj.lower().strip()] = resolved_obj

            result = self.knowledge_graph.add_triple(
                subject=resolved_subj,
                relation=relation,
                obj=resolved_obj,
                confidence=triple.get("final_confidence", triple.get("confidence", 0.7)),
                source=document_id,
                metadata={
                    "evidence": triple.get("supporting_evidence", ""),
                    "verification_status": triple.get("verification_status", "unknown"),
                    "original_subject": raw_subj,
                    "original_object": raw_obj,
                },
            )
            if result is not None:
                added_triples += 1
            else:
                skipped_triples += 1  # Duplicate

        print(f"  Entity resolution: mapped {len(name_to_id)} name variants")
        print(f"  Triples skipped (dup/invalid): {skipped_triples}")
        return added_entities, added_triples

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
        
        entities = []
        for entity_id, entity in self.knowledge_graph.entities.items():
            entities.append({
                "id": entity_id,
                "labels": entity.labels,
                "type": entity.type,
                "metadata": entity.metadata,
            })
        
        triples = []
        for i, triple in enumerate(self.knowledge_graph.triples):
            triples.append({
                "id": f"triple_{i}",
                "subject": triple.subject,
                "relation": triple.relation,
                "object": triple.object,
                "confidence": triple.confidence,
                "source": triple.source,
            })
        
        return {
            "entities": entities,
            "triples": triples,
            "stats": self.get_kg_stats(),
            "integration_stats": self.integration_stats,
        }
