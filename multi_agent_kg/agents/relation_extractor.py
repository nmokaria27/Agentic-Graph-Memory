"""
Relation Extractor Agent.

Implements RHF (Relation-Head-First) multi-stage extraction:
1. Relation Identification: Find relation types present in text
2. Head Entity Binding: Bind relations to head (subject) entities  
3. Tail Entity Binding: Complete triples with tail (object) entities

Features:
- Open-world relation discovery (not limited to predefined types)
- Self-consistency for confidence estimation
- Relation type learning via SharedMemory
- Blackboard voting for novel relations
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING
from dataclasses import dataclass, field
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ValidationError

from multi_agent_kg.schemas_llm import PredictionOut, TripleOut
from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus, CommunicationType
from multi_agent_kg.core.config import LLMConfig

if TYPE_CHECKING:
    from multi_agent_kg.core.deliberation import VoteType


@dataclass
class DiscoveredRelation:
    """A relation type discovered during extraction."""
    name: str
    definition: str
    examples: List[Tuple[str, str, str]] = field(default_factory=list)
    frequency: int = 1
    confidence: float = 0.5
    source_documents: List[str] = field(default_factory=list)


SCIERC_FIXED_TYPE_GUIDE = """ALLOWED RELATION TYPES (use ONLY these exact names).
IMPORTANT: each relation has a fixed HEAD -> TAIL direction. Follow it exactly.

- Used-for: HEAD is a method/tool/system/material, TAIL is the task/application it is used for.
  Direction: (method/tool) -[Used-for]-> (task/application)
  Example: (CNN) -[Used-for]-> (image classification)
  Example: (rule-based parser) -[Used-for]-> (Japanese morphological analysis)

- Part-of: HEAD is a component/subset/stage, TAIL is the whole system or larger entity that contains it.
  Direction: (component) -[Part-of]-> (whole)
  Example: (dictionary lookup) -[Part-of]-> (Amorph)

- Feature-of: HEAD is a feature/property/attribute/characteristic, TAIL is the entity that HAS that feature.
  Direction: (feature/property) -[Feature-of]-> (entity_that_has_the_feature)
  Example: (object shape) -[Feature-of]-> (priori knowledge)
  Example: (robustness) -[Feature-of]-> (Plume system)

- Compare: HEAD and TAIL are two things being directly compared or contrasted (order does not matter semantically; pick one direction consistently).
  Direction: (thing_A) -[Compare]-> (thing_B)
  Example: (our method) -[Compare]-> (baseline)

- Hyponym-of: HEAD is the specific subtype/instance, TAIL is the more general category.
  Direction: (specific_subtype) -[Hyponym-of]-> (general_category)
  Example: (NE items) -[Hyponym-of]-> (proper names)

- Conjunction: HEAD and TAIL are coordinated, listed, or used jointly (either side may come first; pick one direction consistently).
  Direction: (item_A) -[Conjunction]-> (item_B)
  Example: (dictionary lookup) -[Conjunction]-> (rule application)

- Evaluate-for: HEAD is the METRIC or MATERIAL (dataset/corpus) used to evaluate, TAIL is the METHOD or system BEING evaluated. NEVER put the method/system on the HEAD side.
  Direction: (metric_or_dataset) -[Evaluate-for]-> (method_being_evaluated)
  Example: (F1 score) -[Evaluate-for]-> (unlexicalized parser)
  Example: (NEGRA corpus) -[Evaluate-for]-> (unlexicalized parser)
  Example: (repeatability) -[Evaluate-for]-> (interest point detectors)
"""


# Short direction-only hint used in Stage 2 (head binding) and Stage 3 (tail binding).
# Unlike SCIERC_FIXED_TYPE_GUIDE (which is given to Stage 1 for type *selection*),
# the binding stages already know the relation types — they only need to be reminded
# of which side is HEAD and which is TAIL. Keep this compact to avoid blowing up
# prompt length and confusing JSON output.
SCIERC_DIRECTION_HINT = """
HEAD -> TAIL direction reminder:
- Used-for:   (method/tool) -> (task/application)
- Part-of:    (component) -> (whole)
- Feature-of: (feature/property) -> (entity that has it)
- Compare:    (thing_A) -> (thing_B)   [symmetric — pick one order]
- Hyponym-of: (specific subtype) -> (general category)
- Conjunction:(item_A) -> (item_B)     [symmetric — pick one order]
- Evaluate-for: (metric or dataset) -> (method being evaluated)
"""

SCIERC_PAIRWISE_DECISION_RULES = """
SciERC annotation rules for this benchmark:
- Label ONLY direct relations between two listed entities in the same sentence/window.
- Prefer NONE when the sentence merely says both entities are in the same paper, field,
  experiment, dataset description, title, or broad topic.
- Prefer NONE for plausible background knowledge that is not stated by a clear phrase.
- Prefer NONE for generic "related to" or topical association.
- Used-for needs a use/application/function marker: "used for", "applied to",
  "performs", "identifies", "recognizes", "extracts", "predicts", "classifies".
- Feature-of needs an attribute/property marker: "feature", "property",
  "characteristic", "attribute", "uses X as a feature".
- Part-of needs containment/composition: "part of", "component", "includes",
  "consists of", "stage", "module", "subset".
- Hyponym-of needs type/subtype/instance language: "is a", "such as",
  "including", "type of", "class of".
- Evaluate-for needs an evaluation marker: "evaluated on/by", "measured by",
  "score", "accuracy", "dataset/corpus used to evaluate".
- Conjunction is only for explicit coordination/listing: "X and Y", "X, Y, and Z".
- Compare is only for explicit comparison/contrast/baseline language.

Confidence calibration:
- 0.85-1.00: exact marker and clear head/tail direction.
- 0.75-0.84: directly stated but wording is less explicit.
- below 0.75: weak, topical, ambiguous, or plausible but not SciERC-annotatable.
"""

SCIERC_RELATION_ALIASES = {
    "USED_FOR": "Used-for",
    "USES": "Used-for",
    "UTILIZES": "Used-for",
    "APPLIED_TO": "Used-for",
    "APPLIES_TO": "Used-for",
    "FEATURE_OF": "Feature-of",
    "HAS_FEATURE": "Feature-of",
    "PROPERTY_OF": "Feature-of",
    "PART_OF": "Part-of",
    "IS_PART_OF": "Part-of",
    "COMPONENT_OF": "Part-of",
    "HYPONYM_OF": "Hyponym-of",
    "IS_A": "Hyponym-of",
    "TYPE_OF": "Hyponym-of",
    "SUBTYPE_OF": "Hyponym-of",
    "COMPARE": "Compare",
    "COMPARES": "Compare",
    "COMPARED_TO": "Compare",
    "CONJUNCTION": "Conjunction",
    "AND": "Conjunction",
    "COMBINED_WITH": "Conjunction",
    "EVALUATE_FOR": "Evaluate-for",
    "EVALUATED_FOR": "Evaluate-for",
    "EVALUATES": "Evaluate-for",
}


RELATION_IDENTIFICATION_PROMPT = """Identify all relation types present in the following text.

DISCOVER relations from scratch by analyzing the actual text:
- What RELATIONSHIPS are described between entities?
- What CONNECTIONS exist between concepts?
- What ACTIONS or ASSOCIATIONS are mentioned?

DO NOT use predefined relation taxonomies - CREATE types specific to this content.
Extract as MANY distinct relation types as exist — do not merge different relationships into one type.

DOMAIN: {domain}

SUGGESTED TYPES (if any): {suggested_types}

{domain_examples}

TEXT:
{text}

ENTITIES FOUND:
{entities}

Instructions:
1. Look for explicit and implicit relationships between entities
2. Create descriptive relation type names based on what you observe
3. Each relation type should capture a SPECIFIC type of relationship
4. Relation names should be in UPPER_SNAKE_CASE and descriptive

Return:
{{
    "relations_found": [
        {{
            "relation_type": "<DESCRIPTIVE_RELATION_NAME>",
            "definition": "<what this relation means in this context>",
            "count_in_text": <approximate count>,
            "example_text": "<example sentence showing this relation>"
        }}
    ]
}}"""


# Neutral fallback example used only when the DomainClassifier provided no
# document-specific relation examples. Mixed-domain by design.
DEFAULT_RELATION_EXAMPLES = """EXAMPLE:
Text: "Marie Curie founded the Radium Institute in Paris in 1914. She had earlier shared the Nobel Prize with Pierre Curie."
Relations found:
- FOUNDED: "A person established an organization" (e.g., Marie Curie founded Radium Institute)
- LOCATED_IN: "An organization is located in a place" (e.g., Radium Institute located in Paris)
- FOUNDED_IN_YEAR: "An organization was established in a specific year" (e.g., Radium Institute founded in 1914)
- SHARED_AWARD_WITH: "A person shared an award with another person" (e.g., Marie Curie shared Nobel Prize with Pierre Curie)"""


def _format_domain_relation_examples(examples: List[Dict[str, Any]]) -> str:
    """Render DomainClassifier relation examples into the prompt's example format.

    Each example is {"relation_type", "text_span", "subject", "object", "explanation"}.
    Returns "" if nothing usable, so the caller can fall back to defaults.
    """
    lines: List[str] = []
    for ex in examples[:6]:
        if not isinstance(ex, dict):
            continue
        rtype = ex.get("relation_type") or ""
        subj = ex.get("subject") or ""
        obj = ex.get("object") or ""
        if not rtype or not subj or not obj:
            continue
        expl = ex.get("explanation") or ex.get("text_span") or ""
        desc = f' "{expl}"' if expl else ""
        lines.append(f"- {rtype}:{desc} (e.g., {subj} {rtype.lower().replace('_', ' ')} {obj})")
    if not lines:
        return ""
    return "EXAMPLES (relations already observed in this document's domain):\n" + "\n".join(lines)


HEAD_BINDING_PROMPT = """For each relation type, identify the HEAD (subject) entities.

TEXT:
{text}

ENTITIES:
{entities}

RELATION TYPES TO BIND:
{relation_types}
{direction_guide}
For each relation occurrence, identify what entity is the SUBJECT (head) of that relation.

CRITICAL RULES:
- The subject must be copied VERBATIM from the ENTITIES list.
- Copy "head_entity_id" EXACTLY from the "id" field of the matching entity in ENTITIES.
- Never output a generic paraphrase when a concrete entity exists in ENTITIES.
- If no exact entity from ENTITIES expresses the subject, SKIP that occurrence.
- Respect the HEAD -> TAIL direction for each relation type defined above.

Return:
{{
    "head_bindings": [
        {{
            "relation_type": "<relation>",
            "head_entity": "<subject entity text>",
            "head_entity_id": "<entity id if available>",
            "context": "<sentence or phrase containing this>",
            "confidence": <0.0-1.0>
        }}
    ]
}}"""


_STRICT_RELATIONS = os.getenv("STRICT_RELATIONS") == "1"

_STRICT_PRECISION_BLOCK = """

PRECISION OVER RECALL — emit a triple ONLY if BOTH:
  (a) the text contains an explicit linguistic marker that licenses the relation
      (e.g., a verb, preposition, or coordinator: "is", "uses", "founded", "born in",
      "located in", "part of", "evaluated on", "compared with", coordinated NPs); AND
  (b) the head and tail are both syntactically tied to that marker in the same sentence
      (or the immediately preceding sentence via clear coreference).
If either condition fails, SKIP. False triples are penalized far more than missed ones.
Do NOT emit relations based on co-occurrence, plausibility, or domain knowledge.

CONFIDENCE GRADING — use the FULL 0.0-1.0 range:
  - 0.85-1.0: relation is explicitly stated with a clear marker.
  - 0.55-0.84: relation is implied by strong syntactic evidence but no exact marker.
  - 0.0-0.54: weak/uncertain — DO NOT EMIT (skip the triple).
""" if _STRICT_RELATIONS else ""


TAIL_BINDING_PROMPT = """Complete the triples by adding TAIL (object) entities.""" + _STRICT_PRECISION_BLOCK + """


TEXT:
{text}

ENTITIES:
{entities}

HEAD BINDINGS (subject-relation pairs):
{head_bindings}
{direction_guide}
For each head binding, identify what entity is the OBJECT (tail) of that relation.

CRITICAL RULES:
- The OBJECT must be a DIFFERENT entity from the SUBJECT. A triple like (X, relation, X) is INVALID.
- Containment loops are INVALID: do not connect an entity to a near-duplicate substring/superset of itself unless the text explicitly states a type/subtype relation.
- The relation must add a real fact, not restate the subject/object name.
- The object must be an entity from the ENTITIES list or clearly mentioned in the text.
- Copy "subject_id" and "object_id" EXACTLY from the "id" field of the matching entity in ENTITIES whenever the entity appears in the list.
- In benchmark / fixed-schema settings, SUBJECT and OBJECT must be copied VERBATIM from the ENTITIES list. Do not paraphrase entity names.
- If you cannot find a valid, distinct object entity, SKIP that head binding entirely.
- Focus on what the subject ACTS ON, RELATES TO, or AFFECTS — that target is the object.
- Never use generic summaries such as "products of them", "detectors", or "data sources" unless that exact phrase appears as an entity in ENTITIES.
- For conjunctions, connect the explicit coordinated entities themselves, not a generic phrase describing the list.
- Respect the HEAD -> TAIL direction above. For Evaluate-for in particular, the HEAD must be the metric/dataset and the TAIL must be the method being evaluated.

EXAMPLES:
Head binding: {{"relation_type": "REDUCES_BIOMARKER", "head_entity": "Metformin", "context": "Metformin reduces HbA1c levels"}}
Completed triple: {{"subject": "Metformin", "relation": "REDUCES_BIOMARKER", "object": "HbA1c", "confidence": 0.95, "evidence": "Metformin reduces HbA1c levels"}}

Head binding: {{"relation_type": "ASSOCIATED_WITH", "head_entity": "IL-6", "context": "Elevated IL-6 was associated with reduced coronary flow reserve"}}
Completed triple: {{"subject": "IL-6", "relation": "ASSOCIATED_WITH", "object": "coronary flow reserve", "confidence": 0.90, "evidence": "Elevated IL-6 was associated with reduced coronary flow reserve"}}

WRONG (do NOT do this):
{{"subject": "IL-6", "relation": "BIOMARKER_ASSOCIATED_WITH_CONDITION", "object": "IL-6"}} ← INVALID, subject equals object

Return:
{{
    "triples": [
        {{
            "subject": "<head entity>",
            "subject_id": "<head entity id>",
            "relation": "<relation type>",
            "object": "<DIFFERENT tail entity>",
            "object_id": "<tail entity id>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text snippet>"
        }}
    ]
}}"""


PAIRWISE_RELATION_SCORING_PROMPT = """Classify candidate entity pairs into one of the allowed relation types or NONE.

You are working in fixed-schema benchmark mode. Use ONLY these relation labels:
{allowed_relation_types}

{direction_guide}
{decision_rules}

DOCUMENT TEXT:
{text}

CANDIDATE ENTITY PAIRS:
{candidate_pairs}

Instructions:
1. Each candidate pair is already an exact entity pair from the extracted entity list.
2. For each pair, choose exactly one label from the allowed relation types or NONE.
3. Only assign a relation if the text explicitly supports a SciERC annotation.
4. Respect the required HEAD -> TAIL direction for asymmetric relations.
5. For symmetric relations such as Conjunction / Compare, you may keep the presented order.
6. If the best label is uncertain or merely plausible, choose NONE.

Return JSON:
{{
  "predictions": [
    {{
      "pair_index": <index from candidate pairs>,
      "relation": "<allowed relation type or NONE>",
      "confidence": <0.0-1.0>,
      "evidence": "<supporting text span>",
      "direction_rationale": "<short reason>"
    }}
  ]
}}"""


def _normalize_surface(text: str) -> str:
    """Normalize entity surface forms for exact benchmark matching."""
    return " ".join(
        str(text)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _is_degenerate_endpoint_pair(
    subject: Any,
    obj: Any,
    relation: Any = "",
    subject_id: Any = "",
    object_id: Any = "",
) -> bool:
    subj_id = str(subject_id or "").strip().lower()
    obj_id = str(object_id or "").strip().lower()
    if subj_id and obj_id and subj_id == obj_id:
        return True
    subj = _normalize_surface(str(subject or ""))
    obj_norm = _normalize_surface(str(obj or ""))
    if not subj or not obj_norm:
        return False
    if subj == obj_norm:
        return True
    subj_tokens = subj.split()
    obj_tokens = obj_norm.split()
    shorter, longer = (subj, obj_norm) if len(subj_tokens) <= len(obj_tokens) else (obj_norm, subj)
    shorter_tokens = shorter.split()
    if len(shorter_tokens) < 2:
        return False
    relation_norm = str(relation or "").strip().upper().replace("-", "_").replace(" ", "_")
    if relation_norm in {"HYPONYM_OF", "SUBTYPE_OF", "TYPE_OF", "INSTANCE_OF", "PART_OF", "IS_PART_OF", "COMPONENT_OF"}:
        return False
    return re.search(rf"(?<!\w){re.escape(shorter)}(?!\w)", longer) is not None


def _coerce_llm_items(
    result: Any,
    preferred_keys: Tuple[str, ...],
    item_model: Optional[Type[BaseModel]] = None,
) -> List[Dict[str, Any]]:
    """Coerce a raw LLM response into a flat list of dict items.

    The LLM occasionally returns nested lists, the wrong wrapper key, or stringified
    rows instead of a flat list-of-dicts. Without this guard, downstream code that
    calls ``item.get(...)`` crashes with ``'list' object has no attribute 'get'``
    and drops the whole document.

    When ``item_model`` (a Pydantic model) is given, each flattened dict is
    normalized through it — coercing types and giving every item a consistent
    shape — for traceability. The relation/triple models are all-optional and
    ``extra="allow"``, so validation never drops or strips a kept item; it is
    pure normalization, not a filter. The ``except`` is defensive only.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        for key in preferred_keys:
            if key in result:
                result = result[key]
                break
        else:
            return []
    if not isinstance(result, list):
        return []
    flat: List[Dict[str, Any]] = []
    for item in result:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            # Nested list: recurse one level.
            for sub in item:
                if isinstance(sub, dict):
                    flat.append(sub)
        # strings / numbers / None get dropped silently.
    if item_model is None:
        return flat
    normalized: List[Dict[str, Any]] = []
    for item in flat:
        try:
            normalized.append(item_model.model_validate(item).model_dump())
        except ValidationError:
            # All relation/triple fields are optional, so this is unreachable for
            # dict input; kept so a future stricter model can't crash the batch.
            normalized.append(item)
    return normalized


CONNECTIVITY_PASS_PROMPT = """You are given a document and a knowledge graph that was extracted from it.
Many entities are DISCONNECTED (they appear in the document but have no relations in the graph).
Your job is to find relations that connect these disconnected entities to the rest of the graph.

DOCUMENT TEXT:
{text}

DISCONNECTED ENTITIES (no relations yet — find relations for these):
{disconnected_entities}

CONNECTED ENTITIES (already in the graph — can serve as relation partners):
{connected_entities}

KNOWN RELATION TYPES in this graph:
{relation_types}

INSTRUCTIONS:
1. For each disconnected entity, look for ANY relationship it has with connected entities OR other disconnected entities in the document text.
2. You may use the known relation types above OR create new descriptive relation types in UPPER_SNAKE_CASE.
3. Every triple MUST involve at least one disconnected entity.
4. Subject and object MUST be DIFFERENT entities. (X, relation, X) is INVALID.
5. Only extract relations that are supported by the document text.

Return:
{{
    "triples": [
        {{
            "subject": "<entity text>",
            "subject_id": "<entity id if known>",
            "relation": "<RELATION_TYPE>",
            "object": "<entity text>",
            "object_id": "<entity id if known>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text from document>"
        }}
    ]
}}"""


PAIR_COMPLETION_PROMPT = """You are given a document and pairs of entities that BOTH appear in it
near each other, but have NO relation in the extracted knowledge graph yet.

DOCUMENT TEXT:
{text}

CANDIDATE PAIRS (currently unlinked — check each one):
{pairs}

KNOWN RELATION TYPES in this graph:
{relation_types}

INSTRUCTIONS:
1. For each pair, decide whether the document text STATES a relationship between
   the two entities (a verb, preposition, apposition, or explicit phrase).
2. If it does, emit ONE triple for that pair (pick the direction the text supports).
   You may use the known relation types above OR create new descriptive relation
   types in UPPER_SNAKE_CASE.
3. If the text does NOT state a relationship for a pair, OMIT that pair entirely.
   It is normal and correct to omit many pairs — do not invent relations.
4. Subject and object MUST be the two entities of the pair, never the same entity twice.

Return:
{{
    "triples": [
        {{
            "subject": "<entity text>",
            "subject_id": "<entity id>",
            "relation": "<RELATION_TYPE>",
            "object": "<entity text>",
            "object_id": "<entity id>",
            "confidence": <0.0-1.0>,
            "evidence": "<supporting text from document>"
        }}
    ]
}}"""


RELATION_GLEANING_PROMPT = """Review the extracted graph and recover MISSING supported triples.

TEXT:
{text}

ENTITIES:
{entities}

ALLOWED / SUGGESTED RELATION TYPES:
{relation_types}
{direction_guide}

ALREADY EXTRACTED TRIPLES:
{existing_triples}

UNCONNECTED ENTITIES (currently have NO relations — prioritize connecting these):
{unconnected_entities}

Instructions:
1. Add only facts explicitly supported by the text.
2. PRIORITIZE connecting the UNCONNECTED ENTITIES above. Every one of them was
   mentioned in the text for a reason — find the relation that ties it to another
   entity. An entity left with no relation is a failure.
3. LIST & MEMBERSHIP: when the text presents entities as a coordinated list or as
   items/members/parts of a shared parent (e.g. "the committees: A, B, C", "tribes
   such as X and Y", "remedies include P, Q, R"), emit a relation from EACH listed
   item to that shared parent (e.g. MEMBER_OF / PART_OF / HAS_COMMITTEE) — do NOT
   leave the listed items unconnected just because each lacks its own verb.
4. TAXONOMY: for "X is a Y", "X, a kind of Y", "Y such as X" or appositive
   definitions, emit Hyponym-of / IS_A from the specific X to the general Y.
5. ATTRIBUTES: connect dates, values, roles, titles, locations, and capacities to
   the entity they describe (e.g. FOUNDED_IN, LOCATED_IN, HELD_POSITION).
6. Do not duplicate an already extracted triple.
7. In fixed-schema mode, use only the allowed relation labels and copy subject/object from the ENTITIES list.
8. In open-domain mode, relation names may be concise UPPER_SNAKE_CASE predicates, but subject/object must still be concrete mentions from the text.
9. Base every relation on the text — coordination, apposition, and containment are
   valid textual evidence, but pure topical co-occurrence is not.
10. If no supported fact is missing, return an empty list.

Return JSON:
{{
  "missing_triples": [
    {{
      "subject": "<entity text>",
      "subject_id": "<entity id if known>",
      "relation": "<relation label>",
      "object": "<entity/value text>",
      "object_id": "<entity id if known>",
      "confidence": <0.0-1.0>,
      "evidence": "<supporting text span>",
      "rationale": "<why this was missed and why it is supported>"
    }}
  ]
}}"""


class RelationExtractor(BaseAgent):
    """
    Relation Extractor Agent - RHF multi-stage relation extraction.
    
    Pipeline (Relation-Head-First):
    1. Relation Identification: Find what relations exist in text
    2. Head Entity Binding: Bind relations to subject entities
    3. Tail Entity Binding: Complete triples with object entities
    
    Uses SharedMemory to:
    - Track discovered relation types across documents
    - Store extracted triples for cross-reference
    - Post novel relations to blackboard for voting
    
    Uses MessageBus to:
    - Receive domain info and entities
    - Send triples to EvidenceLinker
    - Escalate low-confidence extractions
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
        enable_open_world: bool = True,
        enable_fixed_schema_pairwise: bool = True,
        enable_relation_gleaning: bool = True,
        enable_deterministic_attribute_binding: bool = False,
        fixed_schema_min_triple_confidence: Optional[float] = None,
    ):
        super().__init__(
            name="RelationExtractor",
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
        self.enable_open_world = enable_open_world
        self.enable_fixed_schema_pairwise = enable_fixed_schema_pairwise
        self.enable_relation_gleaning = enable_relation_gleaning
        self.enable_deterministic_attribute_binding = enable_deterministic_attribute_binding
        if fixed_schema_min_triple_confidence is None:
            fixed_schema_min_triple_confidence = float(
                os.getenv("FIXED_SCHEMA_MIN_TRIPLE_CONFIDENCE", "0")
            )
        self.fixed_schema_min_triple_confidence = fixed_schema_min_triple_confidence
        
        # Track discovered relation types
        self.discovered_relations: Dict[str, DiscoveredRelation] = {}
        self.domain_relations: Dict[str, List[str]] = {}

    def _new_funnel_diagnostics(self, document_id: str) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "segments_processed": 0,
            "entities_seen": 0,
            "relations_found": 0,
            "head_bindings": 0,
            "tail_triples": 0,
            "pairwise_pairs_considered": 0,
            "pairwise_positive_predictions": 0,
            "pairwise_triples_added": 0,
            "gleaned_triples_added": 0,
            "invalid_self_refs_filtered": 0,
            "post_alignment_triples": 0,
            "post_dedupe_triples": 0,
            "final_triples": 0,
            "segment_summaries": [],
        }

    def _merge_seed_candidates(
        self,
        all_triples: List[Dict[str, Any]],
        seed_candidates: List[Dict[str, Any]],
        entities: Optional[List[Dict[str, Any]]],
    ) -> int:
        """Fold wide-harvest relationship candidates into the RHF triple pool.

        Candidates use {source, relation, target} shape from the wide call.
        They are aligned to the known entity catalog, deduped against the
        already-extracted triples by normalized (subject, relation, object),
        tagged ``seeded_from_wide`` for provenance, and appended. Returns the
        number of triples actually added.
        """
        def _norm(value: Any) -> str:
            return str(value or "").strip().lower()

        existing = {
            (_norm(t.get("subject")), _norm(t.get("relation")), _norm(t.get("object")))
            for t in all_triples
        }
        seeds: List[Dict[str, Any]] = []
        for cand in seed_candidates:
            if not isinstance(cand, dict):
                continue
            subj, obj = cand.get("source"), cand.get("target")
            rel = str(cand.get("relation", "")).strip()
            if not subj or not obj or not rel:
                continue
            try:
                conf = float(cand.get("confidence", 0.65) or 0.65)
            except (TypeError, ValueError):
                conf = 0.65
            seeds.append({
                "subject": str(subj),
                "relation": rel.upper().replace(" ", "_"),
                "object": str(obj),
                "confidence": conf,
                "metadata": {"seeded_from_wide": True},
            })
        if not seeds:
            return 0
        seeds = self._align_triples_to_known_entities(seeds, entities or [])
        added = 0
        for triple in seeds:
            subj, rel, obj = (
                _norm(triple.get("subject")), _norm(triple.get("relation")), _norm(triple.get("object")),
            )
            if not subj or not obj or subj == obj:
                continue
            if (subj, rel, obj) in existing:
                continue
            existing.add((subj, rel, obj))
            all_triples.append(triple)
            added += 1
        return added

    def _record_funnel_segment(
        self,
        diagnostics: Dict[str, Any],
        *,
        segment_id: Optional[str],
        entities_seen: int,
        relations_found: int,
        head_bindings: int,
        tail_triples: int,
        pairwise_pairs: int,
        pairwise_positives: int,
        pairwise_triples: int,
        gleaned_triples: int,
        invalid_self_refs: int,
        post_align_triples: int,
        post_dedupe_triples: int,
    ) -> None:
        summary = {
            "segment_id": segment_id,
            "entities_seen": entities_seen,
            "relations_found": relations_found,
            "head_bindings": head_bindings,
            "tail_triples": tail_triples,
            "pairwise_pairs_considered": pairwise_pairs,
            "pairwise_positive_predictions": pairwise_positives,
            "pairwise_triples_added": pairwise_triples,
            "gleaned_triples_added": gleaned_triples,
            "invalid_self_refs_filtered": invalid_self_refs,
            "post_alignment_triples": post_align_triples,
            "post_dedupe_triples": post_dedupe_triples,
        }
        diagnostics["segments_processed"] += 1
        for key, value in summary.items():
            if key != "segment_id":
                diagnostics[key] += value
        diagnostics["segment_summaries"].append(summary)

    def _normalize_relation_types(self, relation_types_raw: Any) -> List[str]:
        """Normalize relation types from various formats to List[str]."""
        if not relation_types_raw:
            return []
        
        if not isinstance(relation_types_raw, list):
            return []
        
        normalized = []
        for rt in relation_types_raw:
            if isinstance(rt, dict):
                # Extract 'type' field from dict format
                if "type" in rt:
                    normalized.append(rt["type"])
            elif isinstance(rt, str):
                normalized.append(rt)
        
        return normalized

    def _is_fixed_schema_mode(
        self,
        suggested_types: List[str],
        domain_config: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True when benchmark/fixed-schema extraction should use constrained scoring."""
        if self.enable_open_world:
            return False
        if suggested_types:
            return True
        if domain_config and domain_config.get("relation_types"):
            return True
        return False

    def _canonicalize_fixed_schema_relation(
        self,
        relation: Any,
        allowed_relation_types: List[str],
    ) -> Optional[str]:
        """Return an allowed fixed-schema relation label, or None if invalid.

        GPT-5 often emits harmless label variants such as ``used_for`` or
        ``COMPARES`` even when prompted with the exact SciERC labels. In
        benchmark mode those aliases should be normalized before integration,
        while genuinely out-of-schema labels such as ``Occurs-in`` should be
        dropped instead of entering the KG through governance repair.
        """
        if relation is None:
            return None
        relation_text = str(relation).strip()
        if not relation_text:
            return None
        allowed = [str(label).strip() for label in allowed_relation_types if str(label).strip()]
        allowed_by_norm = {
            label.upper().replace("-", "_").replace(" ", "_"): label
            for label in allowed
        }
        norm = relation_text.upper().replace("-", "_").replace(" ", "_")
        if norm in allowed_by_norm:
            return allowed_by_norm[norm]
        alias = SCIERC_RELATION_ALIASES.get(norm)
        if alias and alias in allowed:
            return alias
        return None

    def _enforce_fixed_schema_relations(
        self,
        triples: List[Dict[str, Any]],
        allowed_relation_types: List[str],
    ) -> List[Dict[str, Any]]:
        if not triples or not allowed_relation_types:
            return triples

        cleaned: List[Dict[str, Any]] = []
        for triple in triples:
            raw_relation = triple.get("relation", "")
            canonical = self._canonicalize_fixed_schema_relation(
                raw_relation,
                allowed_relation_types,
            )
            if canonical is None:
                continue
            if canonical != raw_relation:
                metadata = triple.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["original_relation"] = raw_relation
                metadata["relation_canonicalized"] = True
                triple["metadata"] = metadata
                triple["relation"] = canonical
            cleaned.append(triple)
        return cleaned

    def _filter_fixed_schema_triples_by_confidence(
        self,
        triples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop weak fixed-schema triples before governance admission."""
        threshold = self.fixed_schema_min_triple_confidence
        if threshold <= 0:
            return triples

        kept: List[Dict[str, Any]] = []
        for triple in triples:
            try:
                confidence = float(triple.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence >= threshold:
                kept.append(triple)
        return kept

    def _entity_surface_forms(self, entity: Dict[str, Any]) -> List[str]:
        surfaces: List[str] = []
        if entity.get("text"):
            surfaces.append(str(entity["text"]))
        for label in entity.get("labels", []) or []:
            if label:
                surfaces.append(str(label))
        if entity.get("id"):
            surfaces.append(str(entity["id"]))
        return list(dict.fromkeys(surfaces))

    def _build_local_entity_pairs(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        max_pairs_per_window: int = 18,
    ) -> List[Dict[str, Any]]:
        """Build directed candidate pairs from sentence-local entity co-occurrence."""
        if not text or len(entities) < 2:
            return []

        windows = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+|\n+", text)
            if chunk and chunk.strip()
        ]
        if not windows:
            windows = [text]

        candidate_pairs: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()

        for window_idx, window in enumerate(windows):
            window_lower = window.lower()
            window_entities: List[Dict[str, Any]] = []
            for entity in entities:
                surfaces = self._entity_surface_forms(entity)
                if any(surface.lower() in window_lower for surface in surfaces if surface):
                    window_entities.append(entity)

            if len(window_entities) < 2:
                continue

            added_for_window = 0
            for head in window_entities:
                for tail in window_entities:
                    head_id = str(head.get("id") or head.get("text") or "").strip()
                    tail_id = str(tail.get("id") or tail.get("text") or "").strip()
                    if not head_id or not tail_id or head_id == tail_id:
                        continue
                    key = (head_id.lower(), tail_id.lower(), window_idx)
                    if key in seen:
                        continue
                    candidate_pairs.append(
                        {
                            "pair_index": len(candidate_pairs),
                            "sentence_index": window_idx,
                            "sentence": window,
                            "head_candidate": head.get("text") or head_id,
                            "head_candidate_id": head_id,
                            "head_type": head.get("type", ""),
                            "tail_candidate": tail.get("text") or tail_id,
                            "tail_candidate_id": tail_id,
                            "tail_type": tail.get("type", ""),
                        }
                    )
                    seen.add(key)
                    added_for_window += 1
                    if added_for_window >= max_pairs_per_window:
                        break
                if added_for_window >= max_pairs_per_window:
                    break

        return candidate_pairs

    def _stage_pairwise_relation_scoring(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        allowed_relation_types: List[str],
        stage1_relation_types: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Classify sentence-local entity pairs into the fixed schema or NONE."""
        # In fixed-schema benchmark mode, Stage 1 should not be a hard gate.
        # SciERC is dominated by Used-for; if Stage 1 detects any other label
        # but misses Used-for, constraining pairwise classification to Stage-1
        # labels prevents recovery and destroys recall. Pairwise sees concrete
        # entity pairs, so it should consider the full allowed schema and choose
        # NONE when no relation is supported.
        relation_types = list(dict.fromkeys(allowed_relation_types))
        candidate_pairs = self._build_local_entity_pairs(text, entities)
        stats = {
            "pairwise_pairs_considered": len(candidate_pairs),
            "pairwise_positive_predictions": 0,
            "pairwise_triples_added": 0,
        }
        if not candidate_pairs or not relation_types:
            return [], stats

        triples: List[Dict[str, Any]] = []
        # Batch size 50: 100 candidate pairs as JSON + full text can exceed
        # 20K tokens; with thinking-model max_tokens inflation (3x to 12288)
        # this overflows the 32K context window.
        batch_size = 50
        direction_guide = SCIERC_DIRECTION_HINT

        total_batches = (len(candidate_pairs) + batch_size - 1) // batch_size
        print(f"      Pairwise scoring {len(candidate_pairs)} candidate pairs in {total_batches} batches...")

        for start in range(0, len(candidate_pairs), batch_size):
            batch_num = (start // batch_size) + 1
            if total_batches > 1:
                print(f"        Processing pairwise batch {batch_num}/{total_batches}...")
            batch = candidate_pairs[start:start + batch_size]
            prompt = PAIRWISE_RELATION_SCORING_PROMPT.format(
                allowed_relation_types=", ".join(relation_types + ["NONE"]),
                direction_guide=direction_guide,
                decision_rules=SCIERC_PAIRWISE_DECISION_RULES,
                text=text,
                candidate_pairs=json.dumps(batch, indent=2),
            )
            try:
                result = self.call_llm(
                    prompt=prompt,
                    system_prompt=(
                        "You are an expert at relation classification. "
                        "Classify each directed entity pair conservatively and return valid JSON."
                    ),
                    tier=ModelTier.MEDIUM,
                    max_tokens=4096,
                )
            except Exception as e:
                print(f"  WARNING: Pairwise batch {batch_num}/{total_batches} failed: {e}; skipping batch")
                continue
            predictions = _coerce_llm_items(result, ("predictions",), PredictionOut)
            pair_lookup = {pair["pair_index"]: pair for pair in batch}
            for prediction in predictions:
                relation = str(prediction.get("relation", "")).strip()
                if not relation or relation.upper() == "NONE" or relation not in relation_types:
                    continue
                pair_index = prediction.get("pair_index")
                if pair_index not in pair_lookup:
                    continue
                pair = pair_lookup[pair_index]
                stats["pairwise_positive_predictions"] += 1
                triples.append(
                    {
                        "subject": pair["head_candidate"],
                        "subject_id": pair["head_candidate_id"],
                        "relation": relation,
                        "object": pair["tail_candidate"],
                        "object_id": pair["tail_candidate_id"],
                        "confidence": float(prediction.get("confidence", 0.5)),
                        "evidence": prediction.get("evidence", pair["sentence"]),
                        "metadata": {
                            "pairwise_scored": True,
                            "sentence_index": pair["sentence_index"],
                            "direction_rationale": prediction.get("direction_rationale", ""),
                        },
                    }
                )

        stats["pairwise_triples_added"] = len(triples)
        return triples, stats

    def _dedupe_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dedupe triples, preferring the higher-confidence version."""
        best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for triple in triples:
            if _is_degenerate_endpoint_pair(
                triple.get("subject"),
                triple.get("object"),
                triple.get("relation"),
                triple.get("subject_id"),
                triple.get("object_id"),
            ):
                continue
            key = (
                str(triple.get("subject_id") or triple.get("subject") or "").strip().lower(),
                str(triple.get("relation") or "").strip(),
                str(triple.get("object_id") or triple.get("object") or "").strip().lower(),
            )
            if not all(key):
                continue
            current = best.get(key)
            if current is None or float(triple.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                best[key] = triple
        return list(best.values())

    def run(
        self,
        context: AgentContext,
        segments: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        domain_config: Optional[Dict[str, Any]] = None,
        seed_candidates: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Extract relations using RHF pipeline.
        
        Args:
            context: Processing context
            segments: Document segments
            entities: Extracted entities
            domain_config: Domain configuration
            
        Returns:
            ExtractionResult with extracted triples
        """
        self.stats["calls"] += 1
        
        # Use entities from context if not provided
        entities = entities or context.entities or []
        
        # Get relation types from domain or discovered
        suggested_types = self._get_suggested_relation_types(domain_config)

        # Domain-specific relation examples from the DomainClassifier
        self._domain_relation_examples = (
            (domain_config.get("relation_examples") or []) if domain_config else []
        )

        # Check for domain messages
        if self.message_bus:
            messages = self.receive_messages()
            for msg in messages:
                if msg.comm_type == CommunicationType.INFORM and "relation_types" in msg.content:
                    suggested_types = self._normalize_relation_types(msg.content["relation_types"])
                    if msg.content.get("relation_examples"):
                        self._domain_relation_examples = msg.content["relation_examples"]
        
        # Process segments or full text
        all_triples = []
        low_confidence_triples = []
        new_relations_discovered = []
        identified_relation_types = []
        pairwise_pairs_considered = 0
        pairwise_positive_predictions = 0
        pairwise_triples_added = 0
        gleaned_triples_added = 0
        funnel_diagnostics = self._new_funnel_diagnostics(context.document_id)
        
        texts_to_process = []
        if segments:
            texts_to_process = [(s.get("text", ""), s.get("segment_id")) for s in segments]
        elif context.text:
            texts_to_process = [(context.text, f"{context.document_id}_full")]

        # --- Parallel segment extraction -----------------------------------
        _print_lock = threading.Lock()
        max_workers = int(os.getenv("RELATION_EXTRACT_WORKERS", "8"))
        _single_doc = len(texts_to_process) == 1
        _all_entities = entities  # closure reference

        def _extract_segment_relations(item):
            idx, text, segment_id = item
            if not text or len(text) < 20:
                return idx, segment_id, [], [], [], {"pairwise_pairs_considered": 0, "pairwise_positive_predictions": 0, "pairwise_triples_added": 0}, None

            text_lower = text.lower()
            seg_entities = [
                e for e in _all_entities
                if e.get("source_segment") == segment_id
                or segment_id in (e.get("source_segments") or [])
                or self._entity_has_surface_in_text(e, text_lower)
            ]
            if not seg_entities:
                seg_entities = [e for e in _all_entities if self._entity_has_surface_in_text(e, text_lower)]
            if _single_doc and len(seg_entities) < 2 and len(_all_entities) >= 2:
                seg_entities = _all_entities

            # Stage 1: Relation Identification
            rels_found = self._stage1_identify_relations(text, seg_entities, suggested_types, context.domain)
            rels_found = [r for r in rels_found if isinstance(r, dict) and r.get("relation_type")]

            local_new_relations = [r for r in rels_found if r.get("is_new_type")]
            rel_types = [r["relation_type"] for r in rels_found]

            det_value_triples: List[Dict[str, Any]] = []
            if self.enable_open_world and self.enable_deterministic_attribute_binding:
                det_value_triples = self._extract_answer_bearing_value_triples(text=text, entities=seg_entities)
                det_value_triples.extend(self._extract_typed_attribute_facts(text=text, entities=seg_entities))

            fixed_schema_mode = self._is_fixed_schema_mode(suggested_types, domain_config)
            rel_types_for_recovery = rel_types or (suggested_types if fixed_schema_mode else [])

            _empty_funnel = dict(
                segment_id=segment_id, entities_seen=len(seg_entities),
                relations_found=0, head_bindings=0, tail_triples=0,
                pairwise_pairs=0, pairwise_positives=0, pairwise_triples=0,
                gleaned_triples=0, invalid_self_refs=0, post_align_triples=0, post_dedupe_triples=0,
            )
            # Do not bail when nothing was pre-identified if this is an open-world
            # segment with >=2 entities: the connectivity-aware gleaning pass below
            # can still connect coordinated lists / taxonomic mentions that Stage 1
            # missed. Bailing here is a primary source of orphan entities.
            _can_glean_connectivity = self.enable_open_world and len(seg_entities) >= 2
            if not rel_types_for_recovery and not det_value_triples and not _can_glean_connectivity:
                return idx, segment_id, [], local_new_relations, rel_types, {"pairwise_pairs_considered": 0, "pairwise_positive_predictions": 0, "pairwise_triples_added": 0}, _empty_funnel

            pw_triples: List[Dict[str, Any]] = []
            pw_stats = {"pairwise_pairs_considered": 0, "pairwise_positive_predictions": 0, "pairwise_triples_added": 0}
            if self.enable_fixed_schema_pairwise and fixed_schema_mode:
                pw_triples, pw_stats = self._stage_pairwise_relation_scoring(
                    text=text, entities=seg_entities,
                    allowed_relation_types=suggested_types,
                    stage1_relation_types=rel_types_for_recovery,
                )
                with _print_lock:
                    print(f"  Segment {segment_id}: Pairwise pairs considered = {pw_stats['pairwise_pairs_considered']}")

            head_bindings = self._stage2_head_binding(text, seg_entities, rel_types)
            triples: List[Dict[str, Any]] = []
            if head_bindings:
                triples = self._stage3_tail_binding(text, seg_entities, head_bindings)
            tail_triple_count = len(triples)
            triples.extend(pw_triples)
            triples.extend(det_value_triples)

            gleaned = self._stage4_glean_missing_triples(
                text=text, entities=seg_entities,
                relation_types=(suggested_types if fixed_schema_mode else list(dict.fromkeys(rel_types_for_recovery + suggested_types))),
                existing_triples=triples,
            )
            triples.extend(gleaned)

            if not triples:
                _empty_funnel.update(
                    relations_found=len(rel_types), head_bindings=len(head_bindings),
                    tail_triples=tail_triple_count,
                    pairwise_pairs=pw_stats["pairwise_pairs_considered"],
                    pairwise_positives=pw_stats["pairwise_positive_predictions"],
                    pairwise_triples=pw_stats["pairwise_triples_added"],
                    gleaned_triples=len(gleaned) + len(det_value_triples),
                )
                return idx, segment_id, [], local_new_relations, rel_types, pw_stats, _empty_funnel

            if not self.enable_open_world:
                triples = self._enforce_fixed_schema_relations(triples, suggested_types)
                triples = self._align_triples_to_known_entities(triples, seg_entities)
            triples = self._bind_ids_to_entities(triples, seg_entities)
            post_align_count = len(triples)
            triples = self._dedupe_triples(triples)
            post_dedupe_count = len(triples)

            invalid_self_refs = 0
            final_triples: List[Dict[str, Any]] = []
            for triple in triples:
                subj = (triple.get("subject") or "").strip().lower()
                obj = (triple.get("object") or "").strip().lower()
                subj_id = (triple.get("subject_id") or "").strip().lower()
                obj_id = (triple.get("object_id") or "").strip().lower()
                if _is_degenerate_endpoint_pair(
                    triple.get("subject"),
                    triple.get("object"),
                    triple.get("relation"),
                    triple.get("subject_id"),
                    triple.get("object_id"),
                ):
                    invalid_self_refs += 1
                    continue
                triple["source_segment"] = segment_id
                triple["document_id"] = context.document_id
                try:
                    triple["confidence"] = float(triple.get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    triple["confidence"] = 0.0
                final_triples.append(triple)

            funnel = dict(
                segment_id=segment_id, entities_seen=len(seg_entities),
                relations_found=len(rel_types), head_bindings=len(head_bindings),
                tail_triples=tail_triple_count,
                pairwise_pairs=pw_stats["pairwise_pairs_considered"],
                pairwise_positives=pw_stats["pairwise_positive_predictions"],
                pairwise_triples=pw_stats["pairwise_triples_added"],
                gleaned_triples=len(gleaned) + len(det_value_triples),
                invalid_self_refs=invalid_self_refs,
                post_align_triples=post_align_count,
                post_dedupe_triples=post_dedupe_count,
            )
            return idx, segment_id, final_triples, local_new_relations, rel_types, pw_stats, funnel

        indexed_segs = [(i, text, sid) for i, (text, sid) in enumerate(texts_to_process)]
        seg_rel_results = [None] * len(indexed_segs)  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_extract_segment_relations, item): item[0] for item in indexed_segs}
            for future in as_completed(future_to_idx):
                raw_idx = future_to_idx[future]
                try:
                    seg_rel_results[raw_idx] = future.result()
                except Exception as exc:
                    _, _, sid = indexed_segs[raw_idx]
                    print(f"    ERROR: relation extraction for segment {sid} raised {exc}")
                    seg_rel_results[raw_idx] = (raw_idx, sid, [], [], [], {"pairwise_pairs_considered": 0, "pairwise_positive_predictions": 0, "pairwise_triples_added": 0}, None)

        for result in seg_rel_results:
            if result is None:
                continue
            _idx, segment_id, seg_triples, seg_new_rels, seg_rel_types, pw_stats, funnel = result
            identified_relation_types.extend(rt for rt in seg_rel_types if rt)
            for rel in seg_new_rels:
                new_relations_discovered.append(rel)
                self._register_new_relation(rel, context.document_id)
            if funnel is None:
                continue
            pairwise_pairs_considered += pw_stats.get("pairwise_pairs_considered", 0)
            pairwise_positive_predictions += pw_stats.get("pairwise_positive_predictions", 0)
            pairwise_triples_added += pw_stats.get("pairwise_triples_added", 0)
            for det_val in seg_new_rels:
                if det_val.get("relation_type") == "ANSWER_BEARING_VALUE" and "ANSWER_BEARING_VALUE" not in identified_relation_types:
                    identified_relation_types.append("ANSWER_BEARING_VALUE")
            gleaned_triples_added += funnel.get("gleaned_triples", 0)
            self._record_funnel_segment(funnel_diagnostics, **funnel)
            for triple in seg_triples:
                all_triples.append(triple)
                if triple.get("confidence", 0.0) < self.quality_threshold:
                    low_confidence_triples.append(triple)
        
        # Fold in wide-harvest relationship candidates (hybrid mode). The wide
        # front end already paid for these in its one big call; discarding them
        # made hybrid rebuild pairs from scratch and find fewer (pairR 0.19 vs
        # singlepass 0.23 — MATRIX_REPORT.md verdict 3, leak #2).
        seeded_triples_added = 0
        if seed_candidates:
            seeded_triples_added = self._merge_seed_candidates(
                all_triples, seed_candidates, entities
            )
            if seeded_triples_added:
                print(f"  Wide-harvest seeding added: {seeded_triples_added} candidate triples")
        funnel_diagnostics["seeded_triples_added"] = seeded_triples_added

        # Handle low confidence triples
        print(f"\n[RELATION EXTRACTOR DEBUG]")
        print(f"  Total extracted: {len(all_triples)}")
        print(f"  High confidence (>={self.quality_threshold}): {len(all_triples) - len(low_confidence_triples)}")
        print(f"  Low confidence (<{self.quality_threshold}): {len(low_confidence_triples)}")
        print(f"  New relation types discovered: {len(new_relations_discovered)}")
        print(f"  Gleaned triples added: {gleaned_triples_added}")
        
        if low_confidence_triples:
            self._handle_low_confidence_triples(
                low_confidence_triples,
                context,
            )
        
        # Handle new relation types
        if new_relations_discovered:
            self._handle_new_relations(
                new_relations_discovered,
                context,
            )
        
        # Store results
        if self.shared_memory:
            self._store_triples(all_triples, context.document_id)
        
        # Calculate overall confidence
        if all_triples:
            avg_confidence = sum(t.get("confidence", 0.5) for t in all_triples) / len(all_triples)
        else:
            avg_confidence = 0.0
        funnel_diagnostics["final_triples"] = len(all_triples)
        
        self.log(
            f"Extracted {len(all_triples)} triples, "
            f"{len(new_relations_discovered)} new relation types discovered"
        )
        
        return ExtractionResult(
            items=all_triples,
            confidence=avg_confidence,
            metadata={
                "document_id": context.document_id,
                "low_confidence_count": len(low_confidence_triples),
                "new_relations_discovered": len(new_relations_discovered),
                "relation_types_found": sorted(set(identified_relation_types)),
                "relation_types_used": list(set(t.get("relation", "") for t in all_triples)),
                "suggested_relation_types": suggested_types,
                "pairwise_pairs_considered": pairwise_pairs_considered,
                "pairwise_positive_predictions": pairwise_positive_predictions,
                "pairwise_triples_added": pairwise_triples_added,
                "gleaned_triples_added": gleaned_triples_added,
                "seeded_triples_added": seeded_triples_added,
                "funnel_diagnostics": funnel_diagnostics,
            },
            needs_escalation=len(low_confidence_triples) > 0 or len(new_relations_discovered) > 0,
            escalation_reason=self._get_escalation_reason(low_confidence_triples, new_relations_discovered),
        )

    def _extract_answer_bearing_value_triples(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create simple attribute facts for QA-bearing value entities."""
        if not text or len(entities) < 2:
            return []
        value_types = {"year", "daterange", "date", "quantity", "nationality", "location", "role", "office"}
        values = [
            entity for entity in entities
            if str(entity.get("type", "")).lower() in value_types
            or entity.get("extraction_method") == "deterministic_open_domain_value_harvester"
        ]
        anchors = [entity for entity in entities if entity not in values]
        if not values or not anchors:
            return []

        text_lower = text.lower()
        triples: List[Dict[str, Any]] = []
        for value in values:
            value_text = str(value.get("text") or (value.get("labels") or [""])[0])
            value_key = value_text.lower()
            value_pos = text_lower.find(value_key)
            if value_pos < 0:
                source_text = str(value.get("source_text", ""))
                value_pos = text_lower.find(source_text.lower()[:40]) if source_text else -1
            relation = self._relation_for_answer_value(value, text)
            best_anchor = None
            best_distance = 10**9
            for anchor in anchors:
                if not self._subject_allowed_for_typed_attribute(anchor, value, relation):
                    continue
                surfaces = [
                    anchor.get("text", ""),
                    *(anchor.get("labels") or []),
                    *(anchor.get("mentions") or []),
                ]
                positions = [text_lower.find(str(surface).lower()) for surface in surfaces if surface]
                positions = [position for position in positions if position >= 0]
                if not positions:
                    continue
                distance = min(abs(position - value_pos) for position in positions) if value_pos >= 0 else min(positions)
                if distance < best_distance:
                    best_distance = distance
                    best_anchor = anchor
            if best_anchor is None:
                continue

            triples.append(
                {
                    "subject": best_anchor.get("text") or best_anchor.get("id"),
                    "subject_id": best_anchor.get("id"),
                    "relation": relation,
                    "object": value_text,
                    "object_id": value.get("id"),
                    "confidence": 0.74,
                    "evidence": value.get("source_text") or self._sentence_containing(text, value_text),
                    "metadata": {
                        "answer_bearing_value": True,
                        "deterministic_attribute": True,
                    },
                }
            )
        return triples

    @staticmethod
    def _relation_for_answer_value(value: Dict[str, Any], text: str) -> str:
        value_type = str(value.get("type", "")).lower()
        window = (value.get("source_text") or text or "").lower()
        if value_type == "year":
            return "HAS_YEAR"
        if value_type in {"daterange", "date"}:
            return "HAS_TIMEFRAME"
        if value_type == "quantity":
            if any(term in window for term in ("capacity", "seat", "seated")):
                return "HAS_CAPACITY"
            return "HAS_QUANTITY"
        if value_type in {"role", "office"}:
            return "HELD_POSITION"
        if value_type == "nationality":
            return "HAS_NATIONALITY"
        if value_type == "location":
            return "LOCATED_IN"
        return "HAS_ATTRIBUTE"

    @staticmethod
    def _sentence_containing(text: str, needle: str) -> str:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if needle and needle.lower() in sentence.lower():
                return sentence.strip()
        return text[:300].strip()

    # Generic typed attribute extraction -------------------------------------------
    @staticmethod
    def _entity_surfaces(entity: Dict[str, Any]) -> List[str]:
        surfaces = [
            entity.get("text", ""),
            *(entity.get("labels") or []),
            *(entity.get("mentions") or []),
        ]
        out: List[str] = []
        for s in surfaces:
            s = str(s or "").strip()
            if len(s) >= 2 and s not in out:
                out.append(s)
        return out

    @staticmethod
    def _is_value_like(entity: Dict[str, Any]) -> bool:
        etype = str(entity.get("type", "")).lower()
        return etype in {
            "value", "year", "date", "daterange", "quantity", "nationality",
            "role", "office", "location", "profession", "occupation", "title",
        }

    @staticmethod
    def _relation_for_typed_attribute(entity: Dict[str, Any], evidence: str) -> str:
        etype = str(entity.get("type", "")).lower()
        evidence_lower = evidence.lower()
        if etype == "nationality":
            return "HAS_NATIONALITY"
        if etype in {"role", "profession", "occupation"}:
            return "EXERCISES_PROFESSIONAL_ROLE"
        if etype in {"office", "title"}:
            return "HELD_POSITION"
        if etype in {"date", "daterange"}:
            if "born" in evidence_lower:
                return "BORN_ON_DATE"
            return "HAS_TIMEFRAME"
        if etype == "year":
            return "HAS_YEAR"
        if etype == "location":
            return "LOCATED_IN"
        if etype == "quantity":
            if any(term in evidence_lower for term in ("capacity", "seat", "seated")):
                return "HAS_CAPACITY"
            return "HAS_QUANTITY"
        return "HAS_ATTRIBUTE"

    @staticmethod
    def _looks_like_named_subject(entity: Dict[str, Any]) -> bool:
        """Return true for entities that can plausibly own attribute facts.

        This is intentionally type/shape based, not relation-specific. It
        prevents typed attributes like Nationality or Date from being attached
        to nearby common nouns such as "film producer" just because they are
        closer than the actual subject.
        """
        etype = str(entity.get("type", "")).lower()
        if any(
            token in etype
            for token in (
                "person", "human", "organization", "organisation", "company",
                "institution", "place", "location", "facility", "venue",
                "building", "work", "film", "album", "book", "event",
            )
        ):
            return True

        for surface in RelationExtractor._entity_surfaces(entity):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9'.-]*", surface)
            if not tokens:
                continue
            title_like = sum(1 for token in tokens if token[:1].isupper())
            if title_like >= 2:
                return True
            if title_like == 1 and len(tokens) == 1 and len(tokens[0]) >= 3:
                return True
        return False

    @staticmethod
    def _subject_allowed_for_typed_attribute(
        subject: Dict[str, Any],
        value: Dict[str, Any],
        relation: str,
    ) -> bool:
        if RelationExtractor._is_value_like(subject):
            return False
        if not RelationExtractor._looks_like_named_subject(subject):
            return False

        subject_type = str(subject.get("type", "")).lower()
        value_type = str(value.get("type", "")).lower()

        personish_relations = {
            "HAS_NATIONALITY",
            "EXERCISES_PROFESSIONAL_ROLE",
            "BORN_ON_DATE",
            "HELD_POSITION",
        }
        if relation in personish_relations:
            # These facts should attach to named people or organizations, not
            # nearby role/value entities. Unknown-type named entities are
            # allowed because open-world corpora often type people loosely.
            return (
                any(token in subject_type for token in ("person", "human", "organization", "organisation", "company", "institution"))
                or not subject_type
                or subject_type in {"entity", "unknown", "propernoun", "proper_noun"}
            )

        if relation == "HAS_CAPACITY":
            return value_type == "quantity"

        return True

    def _extract_typed_attribute_facts(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bind typed answer-candidate entities to the nearest subject entity.

        This is deliberately generic: it relies on upstream entity typing
        (Nationality, Role, Date, Quantity, Location, etc.) rather than a
        demonym/profession whitelist. It fixes missed copular/attribute facts
        without baking Wikipedia biography templates into the extractor.
        """
        if not text or not entities:
            return []

        triples: List[Dict[str, Any]] = []
        seen: set = set()

        def add(subject: Dict[str, Any], obj_entity: Dict[str, Any], relation: str, evidence: str):
            obj = str(obj_entity.get("text") or (obj_entity.get("labels") or [""])[0]).strip()
            subject_id = str(subject.get("id") or subject.get("text") or "").strip()
            subject_text = str(subject.get("text") or subject_id).strip()
            key = (subject_id, relation, obj.lower())
            if key in seen or not subject_id or not obj:
                return
            seen.add(key)
            triples.append({
                "subject": subject_text,
                "subject_id": subject_id,
                "relation": relation,
                "object": obj,
                "object_id": obj_entity.get("id"),
                "confidence": 0.78,
                "evidence": evidence.strip(),
                "metadata": {
                    "typed_attribute_fact": True,
                    "deterministic_attribute": True,
                },
            })

        text_lower = text.lower()
        subjects = [entity for entity in entities if not self._is_value_like(entity)]
        values = [entity for entity in entities if self._is_value_like(entity)]
        if not subjects or not values:
            return triples

        for value in values:
            value_surfaces = self._entity_surfaces(value)
            positions = [
                text_lower.find(surface.lower())
                for surface in value_surfaces
                if surface and text_lower.find(surface.lower()) >= 0
            ]
            if not positions:
                continue
            value_pos = min(positions)
            evidence = value.get("source_text") or self._sentence_containing(text, value_surfaces[0])
            relation = self._relation_for_typed_attribute(value, evidence)

            best_subject = None
            best_distance = 10**9
            for subject in subjects:
                if not self._subject_allowed_for_typed_attribute(subject, value, relation):
                    continue
                for surface in self._entity_surfaces(subject):
                    subject_pos = text_lower.find(surface.lower())
                    if subject_pos < 0:
                        continue
                    distance = abs(subject_pos - value_pos)
                    # Prefer subjects occurring before the attribute in the same sentence.
                    if subject_pos <= value_pos:
                        distance -= 50
                    if distance < best_distance:
                        best_distance = distance
                        best_subject = subject

            if best_subject is None:
                continue
            add(
                best_subject,
                value,
                relation,
                evidence,
            )

        return triples

    def _entity_has_surface_in_text(self, entity: Dict[str, Any], text_lower: str) -> bool:
        """Return true when any known entity surface appears in segment text."""
        candidates = [
            entity.get("text", ""),
            entity.get("id", "").replace("_", " "),
            *(entity.get("labels") or []),
            *(entity.get("mentions") or []),
        ]
        for candidate in candidates:
            candidate_lower = str(candidate).lower().strip()
            if len(candidate_lower) >= 2 and candidate_lower in text_lower:
                return True
        return False

    def _get_suggested_relation_types(
        self,
        domain_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Get suggested relation types from domain and discovered."""
        types = []
        
        # From domain config - normalize from dict format
        if domain_config:
            relation_types_raw = domain_config.get("relation_types", [])
            types.extend(self._normalize_relation_types(relation_types_raw))
        
        # From discovered relations (high frequency)
        for name, rel in self.discovered_relations.items():
            if rel.frequency >= 2 and rel.confidence >= 0.6:
                types.append(name)
        
        # From shared memory
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            for mem in memories:
                if "discovered_relations" in mem.content:
                    types.extend(mem.content["discovered_relations"])
                # Also extract relation types from stored triples
                for stored_triple in mem.content.get("triples", []):
                    rel = stored_triple.get("relation", "")
                    if rel and rel not in types:
                        types.append(rel)

        return list(set(types))

    def _stage1_identify_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        suggested_types: List[str],
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Stage 1: Identify relation types in text."""
        entities_str = ", ".join(e.get("text", str(e)) for e in entities)

        # If open_world is disabled, we're in fixed-schema mode — force the types
        if not self.enable_open_world and suggested_types:
            fixed_type_guide = SCIERC_FIXED_TYPE_GUIDE
            prompt = (
                f"Identify which of these SPECIFIC relation types are present in the text.\n\n"
                f"{fixed_type_guide}\n"
                f"Only return relation types from this allowed set:\n"
                + "\n".join(f"- {t}" for t in suggested_types) +
                f"\n\nTEXT:\n{text}\n\n"
                f"ENTITIES FOUND:\n{entities_str}\n\n"
                f"Identify every allowed relation type that is clearly expressed in the text.\n"
                f"Pay special attention to conjunctions such as coordinated pairs, lists, or stages used together.\n"
                f"Do not invent relation types outside the allowed set.\n\n"
                f"Return:\n{{\n"
                f'    "relations_found": [\n'
                f"        {{\n"
                f'            "relation_type": "<one of the allowed types above>",\n'
                f'            "definition": "<what this relation means>",\n'
                f'            "count_in_text": <count>,\n'
                f'            "example_text": "<example>"\n'
                f"        }}\n"
                f"    ]\n"
                f"}}"
            )
        else:
            domain_examples = _format_domain_relation_examples(
                getattr(self, "_domain_relation_examples", []) or []
            ) or DEFAULT_RELATION_EXAMPLES
            prompt = RELATION_IDENTIFICATION_PROMPT.format(
                text=text,
                entities=entities_str,
                suggested_types=", ".join(suggested_types) if suggested_types else "none provided (discover new types)",
                domain=domain or "general",
                domain_examples=domain_examples,
            )
        
        if self.use_self_consistency:
            result, confidence = self.call_llm_with_self_consistency(
                prompt=prompt,
                system_prompt="You are an expert at identifying relations between entities. Be thorough but precise.",
                tier=ModelTier.MEDIUM,
                n_samples=self.n_consistency_samples,
            )
        else:
            result = self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at identifying relations between entities. Be thorough but precise.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )
        
        if isinstance(result, list):
            return result
        return result.get("relations_found", [])

    @staticmethod
    def _entity_catalog_for_prompt(entities: List[Dict[str, Any]]) -> str:
        """Compact id/text/type catalog for binding prompts.

        Keeps prompts small and pushes the model to copy canonical entity ids,
        so triples land on catalog entities instead of free-text paraphrases.
        """
        catalog = [
            {
                "id": e.get("id", e.get("text", "")),
                "text": e.get("text", e.get("id", "")),
                "type": e.get("type", ""),
                "mentions": (e.get("mentions") or [])[:4],
            }
            for e in entities
        ]
        return json.dumps(catalog, indent=2)

    @staticmethod
    def _bind_ids_to_entities(
        triples: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate/fill subject_id and object_id against the entity catalog.

        Runs in all modes (open-world included). An invalid id is cleared so
        downstream resolution doesn't trust a hallucinated id; a missing id is
        filled when the surface text matches an entity's text/id/mentions.
        """
        valid_ids = {str(e.get("id", "")) for e in entities if e.get("id")}
        name_to_id: Dict[str, str] = {}
        for entity in entities:
            eid = str(entity.get("id") or "")
            if not eid:
                continue
            name_to_id[eid.lower().strip()] = eid
            text = str(entity.get("text", "")).lower().strip()
            if text:
                name_to_id[text] = eid
            for mention in entity.get("mentions") or []:
                name_to_id[str(mention).lower().strip()] = eid

        for triple in triples:
            for text_key, id_key in (("subject", "subject_id"), ("object", "object_id")):
                given_id = str(triple.get(id_key) or "").strip()
                if given_id and given_id in valid_ids:
                    continue
                resolved = name_to_id.get(given_id.lower()) if given_id else None
                if resolved is None:
                    surface = str(triple.get(text_key) or "").lower().strip()
                    resolved = name_to_id.get(surface)
                triple[id_key] = resolved or ""
        return triples

    def _stage2_head_binding(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Bind relations to head (subject) entities."""
        if not relation_types:
            return []

        entities_json = self._entity_catalog_for_prompt(entities)
        
        direction_guide = SCIERC_DIRECTION_HINT if not self.enable_open_world else ""
        prompt = HEAD_BINDING_PROMPT.format(
            text=text,
            entities=entities_json,
            relation_types=", ".join(relation_types),
            direction_guide=direction_guide,
        )
        
        result = self.call_llm(
            prompt=prompt,
            system_prompt="You are an expert at identifying subject-relation pairs in text.",
            tier=ModelTier.MEDIUM,
            max_tokens=4096,
        )

        if isinstance(result, list):
            return result
        return result.get("head_bindings", [])

    def _stage3_tail_binding(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        head_bindings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 3: Complete triples with tail (object) entities."""
        if not head_bindings:
            return []
        
        # Process head_bindings in batches to avoid JSON truncation
        batch_size = 15  # Conservative batch size for relation completion
        all_triples = []

        entities_json = self._entity_catalog_for_prompt(entities)
        
        for i in range(0, len(head_bindings), batch_size):
            batch = head_bindings[i:i+batch_size]
            head_bindings_json = json.dumps(batch, indent=2)
            
            direction_guide = SCIERC_DIRECTION_HINT if not self.enable_open_world else ""
            prompt = TAIL_BINDING_PROMPT.format(
                text=text,
                entities=entities_json,
                head_bindings=head_bindings_json,
                direction_guide=direction_guide,
            )
            
            if self.use_self_consistency:
                result, confidence = self.call_llm_with_self_consistency(
                    prompt=prompt,
                    system_prompt="You are an expert at completing relation triples. Be precise about object entities.",
                    tier=ModelTier.MEDIUM,
                    n_samples=self.n_consistency_samples,
                )
                
                # Adjust confidences based on consistency
                triples = _coerce_llm_items(result, ("triples",), TripleOut)
                for t in triples:
                    # Combine LLM confidence with self-consistency
                    t["confidence"] = (t.get("confidence", 0.7) + confidence) / 2
                all_triples.extend(triples)
            else:
                result = self.call_llm(
                    prompt=prompt,
                    system_prompt="You are an expert at completing relation triples. Be precise about object entities.",
                    tier=ModelTier.MEDIUM,
                    max_tokens=4096,
                )
                triples = _coerce_llm_items(result, ("triples",), TripleOut)
                all_triples.extend(triples)

        return all_triples

    def _stage4_glean_missing_triples(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: List[str],
        existing_triples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Recall-oriented second pass for supported facts missed by RHF.

        Connectivity-aware: entities left with no relation after RHF (orphans) are
        surfaced to the LLM so this pass can prioritize connecting them at creation
        time — with full segment text in hand — rather than leaving orphans for a
        degraded post-integration relink. In open-world mode this runs even when no
        relation types were pre-identified, since that is exactly the case where
        coordinated lists / taxonomic mentions get orphaned.
        """
        if not self.enable_relation_gleaning or not text or len(entities) < 2:
            return []
        # In fixed-schema mode we can only glean against known types; skip if none.
        if not relation_types and not self.enable_open_world:
            return []

        # Identify entities that participate in no existing triple.
        connected: Set[str] = set()
        for t in existing_triples:
            for key in ("subject", "object", "subject_id", "object_id"):
                v = t.get(key)
                if v:
                    connected.add(_normalize_surface(str(v)))
        unconnected = [
            e for e in entities
            if _normalize_surface(str(e.get("id", "") or e.get("text", ""))) not in connected
            and _normalize_surface(str(e.get("text", "") or e.get("id", ""))) not in connected
        ]
        # Nothing pre-identified AND everything already connected → nothing to do.
        if not relation_types and not unconnected:
            return []

        prompt = RELATION_GLEANING_PROMPT.format(
            text=text,
            entities=self._entity_catalog_for_prompt(entities),
            relation_types=", ".join(relation_types) if relation_types else "(open-world — propose concise UPPER_SNAKE_CASE predicates)",
            direction_guide=SCIERC_DIRECTION_HINT if not self.enable_open_world else "",
            existing_triples=json.dumps(existing_triples[:40], indent=2),
            unconnected_entities=self._entity_catalog_for_prompt(unconnected) if unconnected else "(none — all entities already have at least one relation)",
        )
        result = self.call_llm(
            prompt=prompt,
            system_prompt=(
                "You are a recall-oriented relation extraction reviewer. "
                "Recover only missing facts that are directly supported by the text."
            ),
            tier=ModelTier.MEDIUM,
            max_tokens=4096,
        )
        triples = _coerce_llm_items(result, ("missing_triples", "triples"), TripleOut)
        cleaned: List[Dict[str, Any]] = []
        allowed = set(relation_types)
        for triple in triples:
            relation = str(triple.get("relation", "")).strip()
            if not relation:
                continue
            if not self.enable_open_world and relation not in allowed:
                continue
            if not triple.get("subject") or not triple.get("object"):
                continue
            if _is_degenerate_endpoint_pair(
                triple.get("subject"),
                triple.get("object"),
                relation,
                triple.get("subject_id"),
                triple.get("object_id"),
            ):
                continue
            triple.setdefault("confidence", 0.65)
            metadata = triple.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["gleaned"] = True
            if triple.get("rationale"):
                metadata["gleaning_rationale"] = triple.get("rationale")
            triple["metadata"] = metadata
            cleaned.append(triple)
        return cleaned

    def _align_triples_to_known_entities(
        self,
        triples: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        In fixed-schema mode, only keep triples whose subject/object can be aligned
        to extracted entities exactly. This avoids generic paraphrases like
        "products of them" becoming benchmark triples.
        """
        if not triples or not entities:
            return triples

        by_id: Dict[str, Dict[str, Any]] = {}
        by_surface: Dict[str, Dict[str, Any]] = {}
        for ent in entities:
            ent_id = str(ent.get("id", "")).strip()
            if ent_id:
                by_id[ent_id.lower()] = ent
            surfaces = set()
            if ent.get("text"):
                surfaces.add(ent["text"])
            for label in ent.get("labels", []) or []:
                surfaces.add(label)
            if ent_id:
                surfaces.add(ent_id)
            for surface in surfaces:
                norm = _normalize_surface(surface)
                if norm:
                    by_surface[norm] = ent

        def resolve(raw_text: str, raw_id: str) -> Optional[Dict[str, Any]]:
            rid = str(raw_id or "").strip().lower()
            if rid and rid in by_id:
                return by_id[rid]
            norm = _normalize_surface(raw_text)
            if norm and norm in by_surface:
                return by_surface[norm]
            return None

        aligned = []
        for triple in triples:
            subj_ent = resolve(triple.get("subject", ""), triple.get("subject_id", ""))
            obj_ent = resolve(triple.get("object", ""), triple.get("object_id", ""))
            if not subj_ent or not obj_ent:
                continue
            subj_id = subj_ent.get("id") or triple.get("subject_id") or triple.get("subject")
            obj_id = obj_ent.get("id") or triple.get("object_id") or triple.get("object")
            if subj_id == obj_id:
                continue
            subj_surface = subj_ent.get("text") or (subj_ent.get("labels") or [subj_id])[0]
            obj_surface = obj_ent.get("text") or (obj_ent.get("labels") or [obj_id])[0]
            triple["subject"] = subj_surface
            triple["subject_id"] = subj_id
            triple["object"] = obj_surface
            triple["object_id"] = obj_id
            aligned.append(triple)
        return aligned

    def _register_new_relation(
        self,
        relation: Dict[str, Any],
        document_id: str,
    ) -> None:
        """Register a newly discovered relation type."""
        name = relation.get("relation_type", "")
        if not name:
            return
        
        if name in self.discovered_relations:
            self.discovered_relations[name].frequency += 1
            self.discovered_relations[name].source_documents.append(document_id)
        else:
            self.discovered_relations[name] = DiscoveredRelation(
                name=name,
                definition=relation.get("definition", ""),
                frequency=1,
                confidence=0.5,
                source_documents=[document_id],
            )

    def _handle_low_confidence_triples(
        self,
        triples: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle low confidence triples via escalation."""
        # Post to blackboard for voting
        for triple in triples[:10]:
            self.post_hypothesis(
                hypothesis={
                    "subject": triple.get("subject"),
                    "relation": triple.get("relation"),
                    "object": triple.get("object"),
                },
                confidence=triple.get("confidence", 0.5),
                evidence=[triple.get("evidence", "")],
            )
        
        # Submit to deliberation for multi-agent voting
        for triple in triples[:10]:  # Limit
            self.submit_for_deliberation(
                hypothesis_type="triple",
                content=triple,
                confidence=triple.get("confidence", 0.5),
                evidence=[triple.get("evidence", "")],
                document_id=context.document_id,
            )
        
        # Escalate to coordinator
        self.escalate_to_coordinator(
            reason="Low confidence relation extractions submitted for deliberation",
            items=triples,
            context={
                "document_id": context.document_id,
                "domain": context.domain,
            },
        )

    def _handle_new_relations(
        self,
        new_relations: List[Dict[str, Any]],
        context: AgentContext,
    ) -> None:
        """Handle newly discovered relation types via deliberation."""
        # Submit new relation types for community voting
        for rel in new_relations:
            self.submit_for_deliberation(
                hypothesis_type="relation_type",
                content={
                    "relation_type": rel.get("relation_type"),
                    "definition": rel.get("definition"),
                },
                confidence=0.6,
                evidence=[context.document_id],
                document_id=context.document_id,
            )
        
        # Store in memory for future reference
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.SEMANTIC,
                content={
                    "discovered_relations": [r.get("relation_type") for r in new_relations],
                    "definitions": {r.get("relation_type"): r.get("definition") for r in new_relations},
                },
            )

    def evaluate_hypothesis_for_vote(
        self,
        hypothesis_content: Dict[str, Any],
        hypothesis_type: str,
        context: Optional[AgentContext] = None,
    ) -> Tuple:
        """
        RelationExtractor's logic for voting on hypotheses.
        
        Can vote on:
        - entity: Abstain (not our specialty)
        - relation: Check if relation type is valid
        - triple: Check if relation makes semantic sense
        - relation_type: Evaluate new relation type proposals
        """
        from multi_agent_kg.core.deliberation import VoteType
        
        if hypothesis_type == "entity":
            # Entities are not our specialty
            return VoteType.ABSTAIN, 0.5, "RelationExtractor focuses on relations"
        elif hypothesis_type == "relation" or hypothesis_type == "triple":
            return self._vote_on_triple(hypothesis_content, context)
        elif hypothesis_type == "relation_type":
            return self._vote_on_relation_type(hypothesis_content, context)
        
        return VoteType.ABSTAIN, 0.5, "RelationExtractor cannot evaluate this hypothesis type"

    def _vote_on_triple(
        self,
        triple: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on a triple hypothesis, consulting memory for known patterns."""
        from multi_agent_kg.core.deliberation import VoteType

        subject = triple.get("subject", "")
        relation = triple.get("relation", "") or triple.get("relation_type", "") or triple.get("predicate", "")
        obj = triple.get("object", "")

        # Basic validation
        if not subject or not relation or not obj:
            return VoteType.REJECT, 0.9, "Triple missing subject, relation, or object"

        # Check if relation type is known from discovered relations
        known_relations = list(self.discovered_relations.keys()) + self.domain_relations.get("general", [])

        # Also check SharedMemory for relation types from stored triples
        if self.shared_memory:
            memories = self.retrieve_from_memory(memory_type=MemoryType.SEMANTIC, limit=10)
            for mem in memories:
                if "discovered_relations" in mem.content:
                    known_relations.extend(mem.content["discovered_relations"])
                for stored_triple in mem.content.get("triples", []):
                    rel = stored_triple.get("relation", "")
                    if rel:
                        known_relations.append(rel)

        known_lower = [r.lower() for r in known_relations]
        if relation.lower() in known_lower:
            return VoteType.ACCEPT, 0.8, f"Known relation type: {relation}"

        # Check for common sense relation patterns
        relation_lower = relation.lower().replace("_", " ")
        common_patterns = ["is a", "works for", "located in", "part of", "born in",
                          "founded", "married to", "has", "owns", "created", "leads",
                          "associated with", "related to", "causes", "treats", "reduces",
                          "increases", "affects", "regulates", "inhibits", "activates"]
        if any(p in relation_lower for p in common_patterns):
            return VoteType.WEAK_ACCEPT, 0.7, "Relation follows common pattern"

        # Domain-specific relations in UPPER_SNAKE_CASE are likely valid
        import re
        if re.match(r'^[A-Z][A-Z0-9_]*$', relation) and len(relation) > 3:
            return VoteType.WEAK_ACCEPT, 0.65, f"Well-formed domain relation: {relation}"

        # Unknown relation - weak reject
        return VoteType.WEAK_REJECT, 0.6, f"Unknown relation type: {relation}"

    def _vote_on_relation_type(
        self,
        relation_type: Dict[str, Any],
        context: Optional[AgentContext],
    ) -> Tuple:
        """Vote on a new relation type proposal."""
        from multi_agent_kg.core.deliberation import VoteType
        
        rel_name = relation_type.get("relation_type", "")
        definition = relation_type.get("definition", "")
        
        if not rel_name:
            return VoteType.REJECT, 0.9, "No relation type name provided"
        
        if not definition:
            return VoteType.WEAK_REJECT, 0.7, "New relation type needs a definition"
        
        # Check if relation already exists
        if rel_name in self.discovered_relations:
            return VoteType.REJECT, 0.8, f"Relation type '{rel_name}' already exists"
        
        # Accept if well-defined
        if len(definition) > 20:
            return VoteType.WEAK_ACCEPT, 0.7, "New relation type with good definition"
        
        return VoteType.WEAK_REJECT, 0.6, "Definition too short for new relation type"

    def _store_triples(
        self,
        triples: List[Dict[str, Any]],
        document_id: str,
    ) -> None:
        """Store extracted triples in memory."""
        self.store_in_memory(
            memory_type=MemoryType.SEMANTIC,
            content={
                "triples": triples,
                "document_id": document_id,
            },
        )

    def _get_escalation_reason(
        self,
        low_confidence: List[Dict[str, Any]],
        new_relations: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Generate escalation reason."""
        reasons = []
        if low_confidence:
            reasons.append(f"{len(low_confidence)} low confidence triples")
        if new_relations:
            reasons.append(f"{len(new_relations)} new relation types")
        return ", ".join(reasons) if reasons else None

    def get_discovered_relations(self) -> Dict[str, DiscoveredRelation]:
        """Get all discovered relation types."""
        return self.discovered_relations

    def extract_connectivity_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        relation_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Connectivity pass: find relations for disconnected entities.

        Args:
            text: Full document text
            entities: All extracted entities
            triples: Already-extracted triples
            relation_types: Known relation types from initial extraction

        Returns:
            List of new triple dicts involving previously disconnected entities
        """
        # Identify connected vs disconnected entities
        connected_ids = set()
        for t in triples:
            subj_id = t.get("subject_id") or t.get("subject", "")
            obj_id = t.get("object_id") or t.get("object", "")
            connected_ids.add(subj_id)
            connected_ids.add(obj_id)

        disconnected = []
        connected = []
        for e in entities:
            eid = e.get("id", e.get("text", ""))
            if eid in connected_ids:
                connected.append(e)
            else:
                disconnected.append(e)

        if not disconnected:
            print("  Connectivity pass: all entities already connected!")
            return []

        print(f"  Connectivity pass: {len(disconnected)} disconnected, {len(connected)} connected entities")

        # Gather known relation types
        if not relation_types:
            relation_types = list(set(
                t.get("relation", "") for t in triples if t.get("relation")
            ))

        # Process disconnected entities in batches
        batch_size = 10
        all_new_triples = []

        batches = [disconnected[i:i + batch_size]
                   for i in range(0, len(disconnected), batch_size)]

        def _connectivity_batch(index, batch):
            # Worker: prompt build + LLM call only (GB-4 fan-out safe).
            disc_str = "\n".join(
                f"- {e.get('id', '?')}: \"{e.get('text', e.get('labels', ['?'])[0] if e.get('labels') else '?')}\" (type: {e.get('type', '?')})"
                for e in batch
            )

            # Find nearest connected entities by text proximity
            nearby_connected = connected[:20]  # Cap to avoid prompt overflow
            conn_str = "\n".join(
                f"- {e.get('id', '?')}: \"{e.get('text', e.get('labels', ['?'])[0] if e.get('labels') else '?')}\" (type: {e.get('type', '?')})"
                for e in nearby_connected
            )

            prompt = CONNECTIVITY_PASS_PROMPT.format(
                text=text[:6000],  # Cap text length
                disconnected_entities=disc_str,
                connected_entities=conn_str,
                relation_types=", ".join(relation_types) if relation_types else "none discovered yet",
            )

            return self.call_llm(
                prompt=prompt,
                system_prompt="You are an expert at discovering relationships between entities in text. Be thorough — find every relationship you can.",
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )

        from multi_agent_kg.core.parallel import map_batches
        for result in map_batches(_connectivity_batch, batches):
            new_triples = _coerce_llm_items(result, ("triples",), TripleOut)

            # Filter self-referencing triples
            for t in new_triples:
                subj = (t.get("subject") or "").strip().lower()
                obj = (t.get("object") or "").strip().lower()
                subj_id = (t.get("subject_id") or "").strip().lower()
                obj_id = (t.get("object_id") or "").strip().lower()
                if subj and obj and not _is_degenerate_endpoint_pair(
                    t.get("subject"),
                    t.get("object"),
                    t.get("relation"),
                    t.get("subject_id"),
                    t.get("object_id"),
                ):
                    t["source"] = "connectivity_pass"
                    all_new_triples.append(t)

        print(f"  Connectivity pass: found {len(all_new_triples)} new triples")
        return all_new_triples

    @staticmethod
    def _pair_completion_candidates(
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        window_chars: int = 300,
        max_pairs: int = 100,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
        """Co-occurring, unlinked entity pairs — GB-3's dominant miss class.

        A candidate pair: both surfaces occur in the text within
        ``window_chars`` of each other AND no existing triple links them
        (either direction, by id or surface). Sorted nearest-first, capped.
        """
        def _norm(s: Any) -> str:
            return re.sub(r"\s+", " ", str(s or "").lower()).strip()

        ntext = _norm(text)
        located = []
        for e in entities:
            surface = e.get("text") or e.get("id", "")
            pos = ntext.find(_norm(surface)) if surface else -1
            if pos >= 0:
                located.append((e, pos))

        linked = set()
        for t in triples:
            ends = frozenset((
                _norm(t.get("subject_id") or t.get("subject")),
                _norm(t.get("object_id") or t.get("object")),
            ))
            linked.add(ends)
            linked.add(frozenset((_norm(t.get("subject")), _norm(t.get("object")))))

        candidates = []
        for i in range(len(located)):
            for j in range(i + 1, len(located)):
                (ea, pa), (eb, pb) = located[i], located[j]
                dist = abs(pa - pb)
                if dist > window_chars:
                    continue
                ka = _norm(ea.get("id") or ea.get("text"))
                kb = _norm(eb.get("id") or eb.get("text"))
                sa = _norm(ea.get("text") or ea.get("id"))
                sb = _norm(eb.get("text") or eb.get("id"))
                if ka == kb or sa == sb:
                    continue
                if frozenset((ka, kb)) in linked or frozenset((sa, sb)) in linked:
                    continue
                candidates.append((ea, eb, dist))
        candidates.sort(key=lambda c: c[2])
        return candidates[:max_pairs]

    def extract_pair_completion_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        relation_types: Optional[List[str]] = None,
        window_chars: int = 300,
        max_pairs: int = 100,
        pairs_per_batch: int = 15,
    ) -> List[Dict[str, Any]]:
        """Pair-completion pass (GB-3, EXP-PAIR-COMPLETE).

        Offline coverage analysis showed 40% of gold pairs are missed while
        BOTH entities are already extracted (28% co-occurring within 300
        chars). This pass shows the model exactly those unlinked co-occurring
        pairs and asks which have a relation STATED in the text — omitting a
        pair is an expected answer, not a failure. Additions are tagged
        ``source=pair_completion`` and flow through the normal verification
        and governance gates downstream.
        """
        candidates = self._pair_completion_candidates(
            text, entities, triples, window_chars=window_chars, max_pairs=max_pairs
        )
        if not candidates:
            print("  Pair completion: no unlinked co-occurring pairs")
            return []

        if not relation_types:
            relation_types = list({
                t.get("relation", "") for t in triples if t.get("relation")
            })

        print(f"  Pair completion: {len(candidates)} candidate pair(s)")
        batches = [candidates[i:i + pairs_per_batch]
                   for i in range(0, len(candidates), pairs_per_batch)]

        def _pair_batch(index, batch):
            # Worker: prompt build + LLM call only (GB-4 fan-out safe).
            pairs_str = "\n".join(
                f"- {a.get('id','?')}: \"{a.get('text', a.get('id','?'))}\"  <-?->  "
                f"{b.get('id','?')}: \"{b.get('text', b.get('id','?'))}\""
                for a, b, _ in batch
            )
            prompt = PAIR_COMPLETION_PROMPT.format(
                text=text[:6000],
                pairs=pairs_str,
                relation_types=", ".join(relation_types) if relation_types else "none discovered yet",
            )
            return self.call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at judging whether a document states a "
                    "relationship between two entities. Be precise; omit pairs "
                    "with no stated relation."
                ),
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
            )

        from multi_agent_kg.core.parallel import map_batches
        all_new: List[Dict[str, Any]] = []
        for result in map_batches(_pair_batch, batches):
            for t in _coerce_llm_items(result, ("triples",), TripleOut):
                subj, obj = t.get("subject"), t.get("object")
                if subj and obj and not _is_degenerate_endpoint_pair(
                    subj, obj, t.get("relation"),
                    t.get("subject_id"), t.get("object_id"),
                ):
                    t["source"] = "pair_completion"
                    all_new.append(t)
        print(f"  Pair completion: found {len(all_new)} new triples")
        return all_new
