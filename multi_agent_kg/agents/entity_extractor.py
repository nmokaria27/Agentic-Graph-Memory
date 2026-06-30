"""
Entity Extractor Agent.

Implements optimized entity extraction pipeline:
1. Combined Extraction: Extract entities with types and verified boundaries in one pass
2. Coreference Resolution: Link mentions to same entity across segments

Features:
- Self-consistency for confidence estimation (optional)
- Cross-document entity resolution via SharedMemory
- Blackboard posting for ambiguous entities
- Iterative refinement with feedback
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import ValidationError

from multi_agent_kg.schemas_llm import CoreferenceOut, EntityExtractionOut
from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Entity
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType


@dataclass
class EntityCandidate:
    """Candidate entity during extraction pipeline."""
    text: str
    start: int = 0
    end: int = 0
    entity_type: Optional[str] = None
    confidence: float = 0.5
    source_segment: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


COMBINED_EXTRACTION_PROMPT = """Extract ALL significant entities from this text. For each entity, provide the EXACT text span as it appears, its type, and your confidence.

DOMAIN: {domain}
{entity_guidance}

EXTRACT entities including:
- Domain concepts, theories, methods, techniques, phenomena, mechanisms
- Medical/scientific conditions, diseases, treatments, clinical measures, biomarkers
- Therapeutic interventions, drugs, procedures, therapies
- Biological processes, pathways, molecular mechanisms
- Clinical outcomes, complications, risk factors, prognostic indicators
- Organizations, institutions, research groups, medical centers
- Researchers, authors, key contributors
- Specialized terminology and technical concepts
- Study cohorts, patient populations, demographic groups
- Measurement tools, instruments, assessment methods, scoring systems
- Important statistical markers (e.g., "HbA1c", "IESS", "CFR") when they represent specific measurements

CRITICAL SPAN RULES:
- Extract the SHORTEST meaningful noun phrase that still identifies the entity.
  GOOD: "morphological analysis"
  BAD:  "morphological analysis problem in Japanese"
- Prefer the core scientific entity over the surrounding event or sentence frame.
  GOOD: "proper nouns"
  BAD:  "Recognition of proper nouns in Japanese text"
- Keep short but meaningful entities.
  VALID: "Amorph", "MET", "NE", "rules", "dictionaries", "constraints"
- If both a short form and a long form are present, extract both when both are meaningful.
  VALID: "NE items" and "Named Entity (NE) items"
- For coordinated lists, extract the individual entities in the list when they are meaningful.
  GOOD: "proper names", "numerical expressions", "temporal expressions"
  BAD:  only the entire long list with no atomic entities
- When a longer phrase contains a clear embedded scientific entity, extract the embedded entity too.
  GOOD: "Recognition of proper nouns in Japanese text" -> also extract "proper nouns"
  GOOD: "morphological analysis problem" -> also extract "morphological analysis"
- Keep qualifiers only when they are necessary to identify the entity.
  GOOD: "constrained optimization scheme"
  BAD:  "the constrained optimization scheme used in our approach"
- Prefer clean entity mentions over explanatory phrases.
  GOOD: "English and Czech newspaper texts"
  BAD:  "annotated English and Czech newspaper texts used for training"

DO NOT EXTRACT:
- Articles, prepositions, pronouns, conjunctions alone
- Generic phrases ("the study", "the results", "the authors", "the data")
- Bare numbers without meaning ("0.85", "38614", "42"); extract numbers when they are answer-bearing dates, measurements, capacities, populations, scores, or time ranges.
- Common adjectives alone ("high", "low", "greater", "significant")

IMPORTANT:
- Err on the side of INCLUSION, but prefer cleaner and shorter entity spans.
- Do NOT skip short entities just because they look simple.
- Tool names, abbreviations, and common scientific noun phrases are valid entities.
- If a sentence names stages, tools, or resources explicitly, extract them individually.

## EXAMPLES

Example 1 (Scientific):
Text: "The constrained optimization scheme deforms a 3-D surface mesh. NE items include proper names, numerical and temporal expressions."
Output:
- "constrained optimization scheme" | Method | 0.95
- "3-D surface mesh" | OtherScientificTerm | 0.90
- "NE items" | OtherScientificTerm | 0.85
- "proper names, numerical and temporal expressions" | OtherScientificTerm | 0.85
- "proper names" | OtherScientificTerm | 0.80
- "temporal expressions" | OtherScientificTerm | 0.80

Example 2 (Medical):
Text: "Metformin reduces HbA1c levels in patients with type 2 diabetes mellitus. The WHO recommends it as first-line therapy."
Output:
- "Metformin" | Method | 0.95
- "HbA1c levels" | OtherScientificTerm | 0.95
- "type 2 diabetes mellitus" | OtherScientificTerm | 0.95
- "WHO" | Material | 0.90
- "first-line therapy" | Method | 0.80

TEXT:
{text}

Return a JSON object:
{{
    "entities": [
        {{
            "text": "<exact entity mention from the text - use COMPLETE phrases>",
            "type": "<entity type>",
            "confidence": <0.0-1.0>
        }}
    ]
}}

Extract ALL entities. Be thorough. Prefer precise entity spans over long descriptive spans."""


OPEN_DOMAIN_VALUE_ENTITY_GUIDANCE = """
OPEN-DOMAIN VALUE ENTITY RULES:
- Extract answer-bearing literal values when they are meaningful facts, including years, date ranges, capacities, counts, populations, measurements, rankings, scores, ages, and time spans.
- Extract offices, job titles, named roles, nationalities, country/city names, creative-work titles, organizations, bands, teams, conferences, arenas, campuses, and aliases.
- Preserve exact short answers that could answer who/what/where/when/how-many questions. This includes full person names, organization names, titles of works, city/country/state names, dates, capacities, and explicit aliases.
- If the text is a title-style encyclopedia paragraph, extract the title subject and its key attributes rather than only broad topical phrases.
- For numeric values, keep the unit or qualifier when present.
  GOOD: "3,677 seated", "from 1986 to 2013", "1999", "9,984 inhabitants"
  BAD: dropping the number because it is a bare value
- Do not extract meaningless isolated numbers, but DO extract numbers/dates that answer who/when/where/how-many questions.
"""


COREFERENCE_PROMPT = """Identify which entity mentions refer to the same real-world entity.

RULES FOR GROUPING:
- Abbreviation = Expansion: "WHO" and "World Health Organization" are the same entity
- Full name = Partial name: "type 2 diabetes mellitus" and "type 2 diabetes" and "T2D" are the same
- Synonyms in context: "metformin" and "Glucophage" when referring to the same drug
- DO NOT group generic phrases: "the organization", "the disease", "the treatment" should NOT be grouped with specific entities

RULES FOR CANONICAL_ID:
- Use clean lowercase_snake_case derived from the canonical name
- Use the FULL descriptive name, not abbreviations (e.g. "type_2_diabetes_mellitus" not "T2DM_1")
- Do NOT append numeric suffixes like _1, _2, _3
- Do NOT append _group suffix
- Keep IDs concise but descriptive (e.g. "empagliflozin", "insulin_resistance", "coronary_flow_reserve")

EXAMPLE:
Entities: ["WHO", "World Health Organization", "T2D", "type 2 diabetes", "type 2 diabetes mellitus", "HbA1c", "glycated hemoglobin"]
Groups:
- canonical_id: "world_health_organization", canonical_name: "World Health Organization", mentions: ["WHO", "World Health Organization"]
- canonical_id: "type_2_diabetes_mellitus", canonical_name: "type 2 diabetes mellitus", mentions: ["T2D", "type 2 diabetes", "type 2 diabetes mellitus"]
- canonical_id: "hba1c", canonical_name: "HbA1c", mentions: ["HbA1c", "glycated hemoglobin"]

TEXT:
{text}

ENTITIES:
{entities_json}

KNOWN ENTITIES FROM PREVIOUS DOCUMENTS:
{known_entities}

Group entities that refer to the same thing. Assign a clean canonical_id (lowercase_snake_case, no _1 or _group suffixes).

Return:
{{
    "entity_groups": [
        {{
            "canonical_id": "<lowercase_snake_case_id>",
            "canonical_name": "<primary human-readable name>",
            "type": "<entity type>",
            "mentions": ["<mention1>", "<mention2>", ...],
            "is_known_entity": <true if matches known entity, false otherwise>
        }}
    ]
}}"""


class EntityExtractor(BaseAgent):
    """
    Entity Extractor Agent - Multi-stage entity extraction with confidence.
    
    Pipeline:
    1. Initial Extraction: Identify candidate entities with LLM
    2. Boundary Refinement: Fix entity boundaries
    3. Type Assignment: Classify entity types based on domain
    4. Coreference Resolution: Link mentions to same entity
    
    Uses SharedMemory to:
    - Get known entities for coreference
    - Store entity aliases for cross-document resolution
    - Post ambiguous entities to blackboard
    
    Uses MessageBus to:
    - Receive domain info from DomainClassifier
    - Send entities to RelationExtractor
    - Escalate low-confidence entities
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
        use_self_consistency: bool = True,
        n_consistency_samples: int = 3,
        enable_deterministic_value_harvesting: bool = False,
    ):
        super().__init__(
            name="EntityExtractor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
            quality_threshold=quality_threshold,
        )
        self.use_self_consistency = use_self_consistency
        self.n_consistency_samples = n_consistency_samples
        self.enable_deterministic_value_harvesting = enable_deterministic_value_harvesting

    def _enforce_strict_schema(
        self,
        entities: List[Dict[str, Any]],
        allowed_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Drop or remap entities whose type is not in the allowed set.

        Handles the two common leak patterns: (1) casing variants like
        OTHERSCIENTIFICTERM vs OtherScientificTerm, (2) invented types like
        PERSON, RESEARCHER, ORGANIZATION that aren't in the fixed schema.
        Normalized-name matches are kept with the canonical casing; everything
        else is dropped so the SciERC F1 eval doesn't see schema leaks.
        """
        if not entities or not allowed_types:
            return entities

        def _norm(s: str) -> str:
            return "".join(ch for ch in str(s).lower() if ch.isalnum())

        allowed_by_norm = {_norm(t): t for t in allowed_types}
        filtered: List[Dict[str, Any]] = []
        dropped = 0
        for ent in entities:
            etype = ent.get("type", "")
            canon = allowed_by_norm.get(_norm(etype))
            if canon is None:
                dropped += 1
                continue
            ent["type"] = canon
            filtered.append(ent)
        if dropped:
            self.log(
                f"Strict schema enforcement dropped {dropped} out-of-schema entities; "
                f"kept {len(filtered)}"
            )
        return filtered

    def _normalize_entity_types(self, entity_types_raw: Any) -> List[str]:
        """Normalize entity types to list of strings, handling dict format from DomainClassifier.

        Also stores full metadata (description, priority) in self._entity_type_metadata
        for use in prompt construction.
        """
        if not entity_types_raw:
            return []

        if not isinstance(entity_types_raw, list):
            return []

        if not hasattr(self, '_entity_type_metadata'):
            self._entity_type_metadata = {}

        normalized = []
        for et in entity_types_raw:
            if isinstance(et, dict):
                type_name = et.get("type", str(et))
                normalized.append(type_name)
                # Preserve full metadata for richer prompt guidance
                self._entity_type_metadata[type_name] = {
                    "description": et.get("description", ""),
                    "priority": et.get("priority", "medium"),
                }
            elif isinstance(et, str):
                normalized.append(et)
            else:
                normalized.append(str(et))

        return normalized

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract entities using multi-stage pipeline.
        
        Args:
            context: Processing context
            segments: Document segments
            domain_config: Domain configuration with entity types
            
        Returns:
            ExtractionResult with extracted entities
        """
        self.stats["calls"] += 1
        
        # Get entity types from domain config or use general
        entity_types_raw = domain_config.get("entity_types", []) if domain_config else []
        entity_types = self._normalize_entity_types(entity_types_raw)
        
        if not entity_types:
            entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT"]
        
        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "entity_types" in msg.content:
                    entity_types = self._normalize_entity_types(msg.content["entity_types"])
        
        # Process each segment
        all_entities = []
        low_confidence_entities = []
        
        texts_to_process = []
        if segments:
            texts_to_process = [(s.get("text", ""), s.get("segment_id")) for s in segments]
        elif context.text:
            texts_to_process = [(context.text, f"{context.document_id}_full")]
        
        # Detect fixed schema mode: if domain config has confidence 0.95 and
        # "FixedSchema" domain, use strict typing
        strict_types = (
            domain_config.get("confidence") == 0.95
            and "Fixed" in domain_config.get("reasoning", "")
        ) if domain_config else False

        total_segments = len(texts_to_process)
        # --- Parallel segment extraction -----------------------------------
        # Each LLM call is independent (no shared write state per segment).
        # We fan out via threads; vLLM handles concurrent requests natively.
        _print_lock = threading.Lock()
        max_workers = int(os.getenv("ENTITY_EXTRACT_WORKERS", "8"))

        def _extract_segment(args):
            i, text, segment_id = args
            if not text:
                return segment_id, []
            with _print_lock:
                print(f"    Segment {i+1}/{total_segments} ({segment_id}) — extracting entities...")
            typed = self._extract_entities_combined(
                text, entity_types, context.domain,
                strict_types=strict_types,
            )
            if not typed:
                with _print_lock:
                    print(f"    WARNING: segment {i+1}/{total_segments} ({segment_id}) returned 0 entities — possible LLM parse failure, data lost for this segment")
            if strict_types and entity_types:
                typed = self._enforce_strict_schema(typed, entity_types)
            if not strict_types and self.enable_deterministic_value_harvesting:
                typed = self._augment_open_domain_entities(text, typed)
            # Annotate segment provenance before returning
            for entity in typed:
                entity["source_segment"] = segment_id
                entity["source_document_id"] = context.document_id
                entity.setdefault("source_text", text[:500])
                try:
                    entity["confidence"] = float(entity.get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    entity["confidence"] = 0.0
            with _print_lock:
                if typed:
                    print(f"      -> {len(typed)} entities extracted")
            return segment_id, typed

        indexed = [(i, text, sid) for i, (text, sid) in enumerate(texts_to_process)]
        # Preserve deterministic output order by sorting futures on segment index
        seg_results: List[tuple] = [None] * len(indexed)  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_extract_segment, item): item[0] for item in indexed}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    seg_results[idx] = future.result()
                except Exception as exc:
                    _, _, sid = indexed[idx]
                    print(f"    ERROR: segment {sid} raised {exc}")
                    seg_results[idx] = (sid, [])

        for _sid, typed in seg_results:
            if typed is None:
                continue
            for entity in typed:
                all_entities.append(entity)
                if entity.get("confidence", 0.0) < self.quality_threshold:
                    low_confidence_entities.append(entity)

        # Stage 4: Coreference Resolution (across all segments)
        known_entities = self._get_known_entities()
        resolved = self._stage4_coreference_resolution(
            context.text,
            all_entities,
            known_entities,
        )

        if strict_types and entity_types:
            resolved = self._enforce_strict_schema(resolved, entity_types)
        
        # Handle low confidence entities
        print(f"\n[ENTITY EXTRACTOR DEBUG]")
        print(f"  Total extracted: {len(all_entities)}")
        print(f"  High confidence (>={self.quality_threshold}): {len(all_entities) - len(low_confidence_entities)}")
        print(f"  Low confidence (<{self.quality_threshold}): {len(low_confidence_entities)}")
        
        if low_confidence_entities:
            self._handle_low_confidence_entities(
                low_confidence_entities,
                context,
            )
        
        # Store results
        if self.shared_memory:
            self._store_entities(resolved, context.document_id)
        
        # Calculate overall confidence
        if resolved:
            avg_confidence = sum(e.get("confidence", 0.5) for e in resolved) / len(resolved)
        else:
            avg_confidence = 0.0
        
        self.log(f"Extracted {len(resolved)} entities (avg confidence: {avg_confidence:.2f})")
        
        return ExtractionResult(
            items=resolved,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_entities),
                "stages_completed": 4,
            },
            needs_escalation=len(low_confidence_entities) > 0,
            escalation_reason=f"{len(low_confidence_entities)} low confidence entities" if low_confidence_entities else None,
        )

    def _extract_entities_combined(
        self,
        text: str,
        entity_types: List[str],
        domain: Optional[str],
        strict_types: bool = False,
    ) -> List[Dict[str, Any]]:
        """Combined extraction: extract entities with types and confidence in one LLM call.

        Args:
            strict_types: If True, force the LLM to use ONLY the provided types
                         (for benchmark evaluation with fixed schemas).
        """
        # Build guidance from entity types if provided by domain classifier
        entity_guidance = ""
        if entity_types:
            entity_types_str = self._normalize_entity_types(entity_types)
            # Use stored metadata for richer per-type descriptions
            metadata = getattr(self, '_entity_type_metadata', {})
            if metadata:
                type_lines = []
                for et in entity_types_str:
                    meta = metadata.get(et, {})
                    desc = meta.get("description", "")
                    if desc:
                        type_lines.append(f"  - {et}: {desc}")
                    else:
                        type_lines.append(f"  - {et}")
                if strict_types:
                    entity_guidance = (
                        "REQUIRED entity types (you MUST use ONLY these types, do NOT invent new types):\n"
                        + "\n".join(type_lines)
                        + "\nEvery entity MUST be assigned one of these exact types."
                    )
                else:
                    entity_guidance = "Suggested entity categories:\n" + "\n".join(type_lines) + "\nYou may use these or create more specific types as needed."
            else:
                if strict_types:
                    entity_guidance = (
                        f"REQUIRED entity types (use ONLY these): {', '.join(entity_types_str)}\n"
                        "Every entity MUST be assigned one of these exact types. Do NOT create new types."
                    )
                else:
                    entity_guidance = f"Suggested entity categories: {', '.join(entity_types_str)}\nYou may use these or create more specific types as needed."

        prompt = COMBINED_EXTRACTION_PROMPT.format(
            text=text,
            entity_guidance=(
                entity_guidance
                if strict_types
                else (entity_guidance + "\n" + OPEN_DOMAIN_VALUE_ENTITY_GUIDANCE).strip()
            ),
            domain=domain or "general",
        )

        if strict_types:
            sys_prompt = (
                "You are an expert at entity extraction. "
                "Use EXACTLY the entity type names provided — do not modify their casing or format."
            )
        else:
            sys_prompt = (
                "You are an expert at entity extraction and typing. "
                "Extract entities with accurate types in UPPER_SNAKE_CASE."
            )

        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt=sys_prompt,
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=sys_prompt,
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
            confidence = 0.7

        # Stamp the call-level confidence onto entities the model emitted without
        # one (preserves prior behavior), then validate against the Pydantic
        # schema which coerces types, clamps confidence, and drops broken items.
        raw_entities = (
            result
            if isinstance(result, list)
            else (result.get("entities") if isinstance(result, dict) else None)
        )
        if isinstance(raw_entities, list):
            for e in raw_entities:
                if isinstance(e, dict) and "confidence" not in e:
                    e["confidence"] = confidence
        try:
            parsed = EntityExtractionOut.model_validate(result)
        except ValidationError as exc:
            print(f"  WARNING: entity extraction output failed validation ({exc}) — treating as 0 entities")
            return []
        return [e.model_dump() for e in parsed.entities]

    def _stage4_coreference_resolution(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        known_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 4: Coreference resolution."""
        if not entities:
            return []
        
        import json
        
        # Batch size 50: with thinking-model max_tokens inflation (3x),
        # 100 entities as JSON (~20K tokens) + 12288 max_tokens overflowed
        # the 32K context window. 50 entities keeps the prompt ~10K tokens.
        batch_size = 50
        all_resolved = []
        
        total_batches = (len(entities) + batch_size - 1) // batch_size
        print(f"    Running coreference resolution on {len(entities)} entities ({total_batches} batches)...")
        
        for i in range(0, len(entities), batch_size):
            batch_num = (i // batch_size) + 1
            print(f"      Processing coref batch {batch_num}/{total_batches}...")
            batch = entities[i:i+batch_size]
            entities_json = json.dumps(batch, indent=2)
            known_json = json.dumps(known_entities[:20], indent=2) if known_entities else "[]"
            
            prompt = COREFERENCE_PROMPT.format(
                text=text[:3000],  # Limit context
                entities_json=entities_json,
                known_entities=known_json,
            )
            
            try:
                result = self.call_llm(
                    prompt=prompt,
                    system_prompt="You are an expert at coreference resolution. Group mentions accurately.",
                    tier=ModelTier.MEDIUM,
                    max_tokens=4096,
                )
            except Exception as e:
                print(f"  WARNING: Coref batch {batch_num}/{total_batches} failed: {e}; skipping batch")
                continue
            
            # Validate the coreference output against the Pydantic schema, which
            # tolerates the list / dict-wrapper shapes, drops malformed groups,
            # and fills defaults. A single bad batch degrades to "no coref this
            # batch" instead of crashing the document (was: 'str' object has no
            # attribute 'get' when a thinking model returned raw CoT prose).
            try:
                groups = [
                    g.model_dump()
                    for g in CoreferenceOut.model_validate(result).entity_groups
                ]
            except ValidationError:
                groups = []
            # Pronouns and generic references that should be dropped
            _PRONOUN_PATTERNS = {
                "it", "its", "they", "them", "their", "this", "that",
                "these", "those", "we", "our", "he", "she", "his", "her",
            }
            for group in groups:
                if not isinstance(group, dict):
                    continue  # skip stray strings/None from a malformed parse
                raw_id = group.get("canonical_id", "")
                canonical_name = group.get("canonical_name", "")
                mentions = group.get("mentions", [])
                etype = group.get("type", "UNKNOWN")

                # Skip UNRESOLVED or pronoun-only entities
                if etype.upper() == "UNRESOLVED":
                    continue
                if canonical_name.lower().strip() in _PRONOUN_PATTERNS:
                    continue
                # Skip if canonical_name is a generic phrase (starts with article + generic noun)
                cn_lower = canonical_name.lower().strip()
                generic_words = {"approach", "method", "system", "technique",
                                 "model", "information", "results", "study",
                                 "findings", "data", "analysis", "set of rules",
                                 "relationship", "association", "support", "community",
                                 "change", "journey", "experience", "voice",
                                 "values", "challenges", "understanding"}
                if cn_lower in generic_words:
                    continue
                # Check if it's truly generic (not a proper name starting with "the")
                if cn_lower.startswith(("our ", "this ", "that ", "these ", "those ", "a set of ", "the ", "a ", "an ")):
                    remaining = cn_lower.split(" ", 1)[-1] if " " in cn_lower else ""
                    if remaining in generic_words or any(remaining.startswith(g) for g in generic_words):
                        continue

                # Clean up the ID: strip _1, _2, _group suffixes the LLM may add
                clean_id = self._clean_entity_id(raw_id, canonical_name)

                # Build labels: include canonical name + all unique mentions
                labels = [canonical_name]
                for m in mentions:
                    if m != canonical_name and m not in labels:
                        labels.append(m)

                source_segments: List[str] = []
                source_document_ids: List[str] = []
                source_texts: List[str] = []
                member_confidences: List[float] = []
                mention_keys = {
                    str(value).lower().strip()
                    for value in [canonical_name, clean_id, raw_id, *mentions]
                    if value
                }
                for original in batch:
                    candidates = [
                        original.get("id", ""),
                        original.get("text", ""),
                        *(original.get("labels") or []),
                        *(original.get("mentions") or []),
                    ]
                    if any(str(candidate).lower().strip() in mention_keys for candidate in candidates if candidate):
                        segment = original.get("source_segment")
                        if segment and segment not in source_segments:
                            source_segments.append(segment)
                        document_id = original.get("source_document_id") or original.get("document_id")
                        if document_id and document_id not in source_document_ids:
                            source_document_ids.append(document_id)
                        source_text = original.get("source_text")
                        if source_text and source_text not in source_texts:
                            source_texts.append(source_text)
                        try:
                            member_confidences.append(float(original.get("confidence", 0) or 0))
                        except (TypeError, ValueError):
                            pass

                # Propagate extraction confidence through coreference instead of
                # flattening every group to 0.7/0.8. Flattening pushed nearly all
                # entities into the deliberation band (<0.85), defeating the
                # high-confidence gate and erasing the extractor's own signal.
                # Use the strongest member mention (coref is a recall operation:
                # merging mentions should not lower the group's confidence) with a
                # small known-entity bonus, clamped to [0, 1].
                if member_confidences:
                    base_confidence = max(member_confidences)
                    if group.get("is_known_entity"):
                        base_confidence = min(1.0, base_confidence + 0.1)
                    resolved_confidence = round(base_confidence, 4)
                else:
                    # No matched mentions (LLM invented the group) — fall back to
                    # the previous conservative defaults.
                    resolved_confidence = 0.8 if group.get("is_known_entity") else 0.7

                resolved_entity = {
                    "id": clean_id,
                    "text": canonical_name,
                    "labels": labels,
                    "type": etype,
                    "mentions": mentions,
                    "confidence": resolved_confidence,
                    "is_known_entity": group.get("is_known_entity", False),
                }
                if source_segments:
                    resolved_entity["source_segment"] = source_segments[0]
                    resolved_entity["source_segments"] = source_segments
                if source_document_ids:
                    resolved_entity["source_document_id"] = source_document_ids[0]
                    resolved_entity["source_document_ids"] = source_document_ids
                if source_texts:
                    resolved_entity["source_text"] = source_texts[0]
                    resolved_entity["source_texts"] = source_texts[:3]
                all_resolved.append(resolved_entity)

                # Register aliases in shared memory
                if self.shared_memory:
                    canonical_id = resolved_entity["id"]
                    for mention in mentions:
                        if mention != canonical_name:
                            self.shared_memory.register_entity_alias(mention, canonical_id)
        
        if not all_resolved:
            print(f"  WARNING: Coreference resolution produced 0 groups from {len(entities)} entities — falling back to raw entities")
            return entities

        # Coref runs in batches of `batch_size`; the same canonical entity can
        # surface in more than one batch and produce duplicate groups that the
        # per-batch LLM never sees together. Merge groups sharing the same
        # canonical id so the KG gets one node, not one-per-batch.
        merged = self._merge_resolved_by_id(all_resolved)
        if len(merged) < len(all_resolved):
            print(f"    Merged {len(all_resolved) - len(merged)} cross-batch duplicate coref groups")
        return merged

    @staticmethod
    def _merge_resolved_by_id(resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse resolved coref groups that share a canonical id.

        Union mentions/labels/sources, keep the max confidence, and OR the
        known-entity flag. Order is preserved by first appearance.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for entity in resolved:
            key = entity.get("id") or entity.get("text", "")
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(entity)
                order.append(key)
                continue
            # Union list-valued fields without duplicates, preserving order.
            for field_name in ("labels", "mentions", "source_segments",
                                "source_document_ids", "source_texts"):
                combined = list(existing.get(field_name) or [])
                for value in (entity.get(field_name) or []):
                    if value not in combined:
                        combined.append(value)
                if combined:
                    existing[field_name] = combined
            # Keep first non-empty singular provenance fields.
            for field_name in ("source_segment", "source_document_id", "source_text"):
                if not existing.get(field_name) and entity.get(field_name):
                    existing[field_name] = entity[field_name]
            existing["confidence"] = max(
                float(existing.get("confidence", 0) or 0),
                float(entity.get("confidence", 0) or 0),
            )
            existing["is_known_entity"] = bool(
                existing.get("is_known_entity") or entity.get("is_known_entity")
            )
        return [merged[key] for key in order]

    def _augment_open_domain_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add deterministic answer-bearing value/name candidates.

        LLM entity extraction often drops values that are crucial for QA
        (years, date ranges, capacities, official titles). In open-world mode,
        keeping these as low-cost candidates improves the graph's ability to
        answer factual questions without hard-coding any benchmark labels.
        """
        augmented = list(entities)
        seen = {
            self._normalise_entity_surface(candidate.get("text", ""))
            for candidate in augmented
            if candidate.get("text")
        }

        patterns: List[Tuple[str, str]] = [
            (
                r"\b(?:born|died|opened|founded|released|launched|established)\s+(?:on\s+)?(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{4})\b",
                "Date",
            ),
            (r"\bfrom\s+(?:18|19|20)\d{2}\s+to\s+(?:18|19|20)\d{2}\b", "DateRange"),
            (
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+(?:18|19|20)\d{2}\b",
                "Date",
            ),
            (
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:18|19|20)\d{2}\b",
                "Date",
            ),
            (r"\b(?:18|19|20)\d{2}\b", "Year"),
            (
                r"\b\d{1,3}(?:,\d{3})+(?:\s+(?:seated|seats|people|inhabitants|spectators|capacity))?\b",
                "Quantity",
            ),
            (
                r"\b\d{1,3}(?:\.\d+)?\s*(?:km|mi|miles|kilometers|metres|meters|kg|lb|tons|million|billion|percent|%)\b",
                "Quantity",
            ),
            (
                r"\b[A-Z][A-Za-z]+(?:\s+(?:of|the|and|for|in|on|at|de|van|von|[A-Z][A-Za-z]+)){1,6}(?:,\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)?\b",
                "NamedEntity",
            ),
        ]

        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                surface = match.group(0).strip()
                if len(surface) < 3:
                    continue
                if surface.lower() in {"title", "return only", "json object"}:
                    continue
                key = self._normalise_entity_surface(surface)
                if not key or key in seen:
                    continue
                seen.add(key)
                augmented.append(
                    {
                        "text": self._clean_answer_bearing_surface(surface, entity_type),
                        "labels": [self._clean_answer_bearing_surface(surface, entity_type)],
                        "type": entity_type,
                        "confidence": 0.72,
                        "source_text": self._window_around_span(text, match.start(), match.end()),
                        "extraction_method": "deterministic_open_domain_value_harvester",
                    }
                )
        return augmented

    @staticmethod
    def _clean_answer_bearing_surface(surface: str, entity_type: str) -> str:
        cleaned = surface.strip()
        if entity_type == "Date":
            cleaned = re.sub(r"^(born|died|opened|founded|released|launched|established)\s+(on\s+)?", "", cleaned, flags=re.I)
        return cleaned.strip()

    @staticmethod
    def _normalise_entity_surface(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _window_around_span(text: str, start: int, end: int, radius: int = 180) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right].strip()

    @staticmethod
    def _clean_entity_id(raw_id: str, canonical_name: str) -> str:
        """Clean up entity IDs by removing _1, _group suffixes and normalizing format.

        If the raw_id is empty or purely numeric, derive from canonical_name instead.
        """
        import re

        # If empty or purely numeric, derive from name
        if not raw_id or re.fullmatch(r'\d+', raw_id):
            if canonical_name:
                return re.sub(r'[^a-z0-9]+', '_', canonical_name.lower()).strip('_')
            return raw_id

        clean = raw_id.strip()

        # Strip trailing _1, _2, ... _N suffixes (but not meaningful ones like "c3a")
        # Only strip if the part before the suffix is 3+ chars (avoids stripping "c3" from "c3_1")
        clean = re.sub(r'(?<=\w{3})_\d+$', '', clean)

        # Strip trailing _group suffix
        clean = re.sub(r'_group$', '', clean)

        # Normalize: lowercase, replace spaces/special chars with underscores
        clean = re.sub(r'[^a-z0-9_]', '_', clean.lower())
        clean = re.sub(r'_+', '_', clean).strip('_')

        return clean if clean else raw_id

    def _get_known_entities(self) -> List[Dict[str, Any]]:
        """Get known entities from memory, knowledge graph, and entity aliases."""
        known = []
        seen_ids = set()

        # From knowledge graph
        if self.knowledge_graph:
            for entity_id, entity in list(self.knowledge_graph.entities.items())[:50]:
                known.append({
                    "id": entity_id,
                    "text": entity.labels[0] if entity.labels else entity_id,
                    "type": entity.type,
                })
                seen_ids.add(entity_id)

        # From shared memory — entities stored by previous extraction runs
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=20)
            for mem in memories:
                for ent in mem.content.get("entities", [])[:10]:
                    eid = ent.get("id", ent.get("text", ""))
                    if eid not in seen_ids:
                        known.append(ent)
                        seen_ids.add(eid)

            # Include entity aliases for better coreference
            for alias, canonical_id in self.shared_memory.entity_aliases.items():
                if canonical_id not in seen_ids:
                    known.append({
                        "id": canonical_id,
                        "text": alias,
                        "type": "ALIAS",
                    })

        return known

    def _handle_low_confidence_entities(
        self,
        entities: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle low confidence entities via deliberation."""
        # Submit to deliberation for multi-agent voting
        for entity in entities[:10]:  # Limit to avoid flooding
            self.submit_for_deliberation(
                hypothesis_type="entity",
                content=entity,
                confidence=entity.get("confidence", 0.5),
                evidence=[entity.get("source_segment", "")],
                document_id=context.document_id,
            )
        
        # Also escalate to coordinator for awareness
        self.escalate_to_coordinator(
            reason="Low confidence entity extractions submitted for deliberation",
            items=entities,
            context={
                "document_id": context.document_id,
                "domain": context.domain,
            },
        )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        EntityExtractor's logic for voting on hypotheses.
        
        Can vote on:
        - entity: Check if it looks like a valid entity
        - relation: Check if entities exist
        - triple: Check if entities are valid
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        if hypothesis_type == "entity":
            # Evaluate entity hypothesis
            return self._vote_on_entity(hypothesis_content, context)
        elif hypothesis_type == "relation":
            # Check if the entities in the relation exist
            return self._vote_on_relation_entities(hypothesis_content, context)
        elif hypothesis_type == "triple":
            # Check if subject/object are valid entities
            return self._vote_on_triple_entities(hypothesis_content, context)
        
        return VoteType.ABSTAIN, 0.5, "EntityExtractor cannot evaluate this hypothesis type"

    def _vote_on_entity(
        self,
        entity: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on an entity hypothesis."""
        import re
        from multi_agent_kg.core.deliberation import VoteType

        entity_text = entity.get("text", entity.get("name", ""))
        entity_type = entity.get("type", "")

        # Basic validation checks
        if not entity_text or len(entity_text) < 2:
            return VoteType.REJECT, 0.9, "Entity text too short or empty"

        if len(entity_text) > 100:
            return VoteType.WEAK_REJECT, 0.7, "Entity text suspiciously long"

        # Validate entity type: accept any UPPER_SNAKE_CASE type,
        # only reject if malformed (lowercase, single char, etc.)
        if entity_type:
            if len(entity_type) < 2:
                return VoteType.WEAK_REJECT, 0.6, f"Entity type too short: {entity_type}"
            if not re.match(r'^[A-Z][A-Z0-9_]*$', entity_type):
                return VoteType.WEAK_REJECT, 0.6, f"Malformed entity type: {entity_type}"

        # Check SharedMemory for known entities
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            for mem in memories:
                known = mem.content.get("entities", [])
                for known_entity in known:
                    known_text = known_entity.get("text", known_entity.get("name", ""))
                    if known_text and known_text.lower() == entity_text.lower():
                        return VoteType.ACCEPT, 0.85, f"Entity '{entity_text}' found in memory"

        # Check if it looks like a real entity (capitalized, etc.)
        if entity_text[0].isupper():
            return VoteType.WEAK_ACCEPT, 0.7, "Entity appears to be properly capitalized"

        # Default to weak accept if nothing wrong
        return VoteType.WEAK_ACCEPT, 0.6, "Entity passes basic validation"

    def _vote_on_relation_entities(
        self,
        relation: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on whether a relation's entities are valid."""
        from multi_agent_kg.core.deliberation import VoteType
        
        subject = relation.get("subject", {})
        obj = relation.get("object", {})
        
        # Check if entities look valid
        subject_text = subject.get("text", "") if isinstance(subject, dict) else str(subject)
        obj_text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
        
        if not subject_text or not obj_text:
            return VoteType.REJECT, 0.9, "Missing subject or object entity"
        
        return VoteType.WEAK_ACCEPT, 0.6, "Entities in relation appear valid"

    def _vote_on_triple_entities(
        self,
        triple: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on whether a triple's entities are valid, consulting memory."""
        from multi_agent_kg.core.deliberation import VoteType

        subject = triple.get("subject", "")
        obj = triple.get("object", "")

        if not subject or not obj:
            return VoteType.REJECT, 0.9, "Triple missing subject or object"

        # Check SharedMemory for known entities
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            known_texts = set()
            for mem in memories:
                for ent in mem.content.get("entities", []):
                    known_texts.add((ent.get("text", ent.get("name", ""))).lower())
            subj_known = subject.lower() in known_texts
            obj_known = obj.lower() in known_texts
            if subj_known and obj_known:
                return VoteType.ACCEPT, 0.85, "Both entities found in memory"
            if subj_known or obj_known:
                return VoteType.WEAK_ACCEPT, 0.75, "One entity found in memory"

        return VoteType.WEAK_ACCEPT, 0.6, "Triple entities appear valid"

    def _store_entities(
        self,
        entities: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store extracted entities in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "entities": entities,
                "document_id": document_id,
            },
        )
        
        # Add context for each entity
        for entity in entities:
            self.shared_memory.add_entity_context(
                entity_id=entity.get("id", entity.get("text", "")),
                context={
                    "document_id": document_id,
                    "type": entity.get("type"),
                    "confidence": entity.get("confidence"),
                },
            )
