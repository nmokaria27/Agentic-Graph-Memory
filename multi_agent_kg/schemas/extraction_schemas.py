"""
Pydantic v2 schemas for all LLM responses in the KG extraction pipeline.

These schemas serve two purposes:
1. Constrained decoding: Passed to Ollama's `format` parameter to guarantee
   structurally valid JSON output via GBNF grammar constraints.
2. Validation: Pydantic validators enforce semantic constraints (confidence
   ranges, non-empty strings, etc.) after parsing.

Design rules for Ollama/llama.cpp GBNF compatibility:
- Keep schemas flat (avoid deep nesting beyond 2 levels)
- Use enums for constrained string fields
- Avoid mixing `properties` with `anyOf`/`oneOf`
- Always set `additionalProperties: false` (Pydantic v2 default)
- Use integer min/max only (not float min/max -- llama.cpp limitation)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Entity Extraction (consolidated: initial + type assignment) ───────

class ExtractedEntity(BaseModel):
    """A single entity extracted from text with its type."""
    text: str = Field(description="Exact entity mention from the text")
    type: str = Field(description="Entity type in UPPER_SNAKE_CASE")
    confidence: float = Field(
        default=0.7,
        description="Confidence score between 0 and 1",
    )
    type_reasoning: str = Field(
        default="",
        description="Brief reason for the assigned type",
    )


class EntityExtractionResponse(BaseModel):
    """Response from the consolidated entity extraction prompt."""
    entities: list[ExtractedEntity] = Field(default_factory=list)


# ── Type Assignment (used if running type assignment separately) ──────

class TypedEntity(BaseModel):
    """Entity with type assignment result."""
    text: str = Field(description="Entity text")
    type: str = Field(description="Assigned type in UPPER_SNAKE_CASE")
    type_confidence: float = Field(default=0.7)
    type_reasoning: str = Field(default="")


class TypeAssignmentResponse(BaseModel):
    """Response from the type assignment stage."""
    entities: list[TypedEntity] = Field(default_factory=list)


# ── Coreference / Entity Resolution ──────────────────────────────────

class CoreferenceGroup(BaseModel):
    """A group of entity mentions that refer to the same real-world entity."""
    canonical_id: str = Field(description="Unique identifier for this entity")
    canonical_name: str = Field(description="Primary/preferred name")
    type: str = Field(description="Entity type")
    mentions: list[str] = Field(default_factory=list)
    is_known_entity: bool = Field(default=False)


class CoreferenceResponse(BaseModel):
    """Response from coreference resolution."""
    entity_groups: list[CoreferenceGroup] = Field(default_factory=list)


class EntityResolutionPair(BaseModel):
    """Result of comparing two entities for potential merge."""
    same_entity: bool = Field(description="Whether both refer to the same real-world entity")
    canonical_name: str = Field(description="Preferred name if they are the same entity")
    reason: str = Field(default="", description="Reason for the decision")


class EntityResolutionResponse(BaseModel):
    """Response from entity resolution pair comparison."""
    pairs: list[EntityResolutionPair] = Field(default_factory=list)


# ── Relation Extraction ──────────────────────────────────────────────

class DiscoveredRelation(BaseModel):
    """A relation type discovered in the text."""
    relation_type: str = Field(description="Relation name in UPPER_SNAKE_CASE")
    definition: str = Field(default="", description="What this relation means")
    count_in_text: int = Field(default=1)
    example_text: str = Field(default="")


class RelationIdentificationResponse(BaseModel):
    """Response from relation identification."""
    relations_found: list[DiscoveredRelation] = Field(default_factory=list)


class ExtractedTriple(BaseModel):
    """A complete (subject, relation, object) triple with evidence."""
    subject: str = Field(description="The source/head entity")
    relation: str = Field(description="The relationship type")
    object: str = Field(description="The target/tail entity")
    confidence: float = Field(default=0.7)
    evidence: str = Field(default="", description="Supporting text snippet")


class TripleExtractionResponse(BaseModel):
    """Response from the consolidated triple extraction prompt."""
    triples: list[ExtractedTriple] = Field(default_factory=list)


# ── Evidence Linking ─────────────────────────────────────────────────

class LinkedTriple(BaseModel):
    """A triple linked to its supporting evidence in the source text."""
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation type")
    object: str = Field(description="Object entity")
    confidence: float = Field(default=0.7)
    evidence_text: str = Field(default="", description="Exact supporting quote")
    evidence_type: str = Field(default="explicit", description="explicit, implicit, or inferred")
    evidence_strength: float = Field(default=0.7)


class EvidenceLinkingResponse(BaseModel):
    """Response from evidence linking."""
    linked_triples: list[LinkedTriple] = Field(default_factory=list)


# ── Verification ─────────────────────────────────────────────────────

class VerifiedTriple(BaseModel):
    """A triple after verification against source text."""
    subject: str
    relation: str
    object: str
    verified: bool = Field(default=True)
    verification_status: str = Field(
        default="verified",
        description="verified, partial, rejected, or hallucinated",
    )
    final_confidence: float = Field(default=0.7)
    supporting_evidence: str = Field(default="")
    rejection_reason: str = Field(default="")


class VerificationSummary(BaseModel):
    """Summary counts from verification."""
    total: int = Field(default=0)
    verified: int = Field(default=0)
    partial: int = Field(default=0)
    rejected: int = Field(default=0)
    hallucinated: int = Field(default=0)


class VerificationResponse(BaseModel):
    """Response from extraction verification."""
    verified_triples: list[VerifiedTriple] = Field(default_factory=list)
    verification_summary: Optional[VerificationSummary] = None


# ── Validation (iterative refinement) ────────────────────────────────

class ValidationIssue(BaseModel):
    """An issue found during validation."""
    item_type: str = Field(description="entity or triple")
    item_text: str = Field(default="")
    issue: str = Field(description="Description of the problem")
    severity: str = Field(default="medium", description="low, medium, or high")
    suggestion: str = Field(default="")


class ValidationResponse(BaseModel):
    """Response from extraction validation."""
    is_valid: bool = Field(default=True)
    quality_score: float = Field(default=0.7)
    issues: list[ValidationIssue] = Field(default_factory=list)
    refined_entities: list[ExtractedEntity] = Field(default_factory=list)
    refined_triples: list[ExtractedTriple] = Field(default_factory=list)


# ── Domain Analysis ──────────────────────────────────────────────────

class EntityTypeSpec(BaseModel):
    """Specification for an entity type in the domain schema."""
    type: str = Field(description="Type name in UPPER_SNAKE_CASE")
    description: str = Field(default="")
    priority: str = Field(default="medium", description="high, medium, or low")
    examples_from_text: list[str] = Field(default_factory=list)


class RelationTypeSpec(BaseModel):
    """Specification for a relation type in the domain schema."""
    type: str = Field(description="Relation name in UPPER_SNAKE_CASE")
    description: str = Field(default="")
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    priority: str = Field(default="medium")
    example_from_text: str = Field(default="")


class ExtractionParameters(BaseModel):
    """Parameters guiding extraction depth and strategy."""
    complexity: str = Field(default="medium")
    knowledge_density: str = Field(default="moderate")
    recommended_chunk_size: int = Field(default=512)
    requires_coreference: bool = Field(default=True)
    has_temporal_relations: bool = Field(default=False)
    has_hierarchical_entities: bool = Field(default=False)


class FewShotExample(BaseModel):
    """A few-shot example from the document."""
    text_span: str = Field(default="")
    entity: str = Field(default="")
    entity_type: str = Field(default="")
    explanation: str = Field(default="")


class DomainAnalysisResponse(BaseModel):
    """Response from domain classification and schema generation."""
    primary_domain: str = Field(default="General")
    sub_domains: list[str] = Field(default_factory=list)
    domain_description: str = Field(default="")
    confidence: float = Field(default=0.5)
    reasoning: str = Field(default="")
    key_indicators: list[str] = Field(default_factory=list)
    entity_types: list[EntityTypeSpec] = Field(default_factory=list)
    relation_types: list[RelationTypeSpec] = Field(default_factory=list)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    extraction_parameters: Optional[ExtractionParameters] = None


# ── Critic-Corrector (Phase 5) ───────────────────────────────────────

class CriticIssue(BaseModel):
    """A specific issue identified by the critic agent."""
    item_type: str = Field(description="entity or triple")
    item_text: str = Field(default="", description="The problematic item")
    issue_type: str = Field(
        description="factual_error, missing_info, hallucination, type_mismatch, or redundancy",
    )
    description: str = Field(description="Specific description of the problem")
    severity: str = Field(default="medium", description="low, medium, or high")
    suggested_fix: str = Field(default="")


class CriticFeedback(BaseModel):
    """Complete feedback from the critic agent."""
    issues: list[CriticIssue] = Field(default_factory=list)
    entities_ok: bool = Field(default=True, description="Whether entities pass review overall")
    triples_ok: bool = Field(default=True, description="Whether triples pass review overall")
    overall_quality: float = Field(default=0.7, description="Overall quality score 0-1")


class CorrectedItem(BaseModel):
    """A corrected entity or triple from the corrector agent."""
    original: str = Field(description="Original item text")
    corrected: str = Field(description="Corrected item text")
    action: str = Field(description="fix, remove, or keep")
    reason: str = Field(default="")


class CorrectorResponse(BaseModel):
    """Response from the corrector agent."""
    corrected_entities: list[ExtractedEntity] = Field(default_factory=list)
    corrected_triples: list[ExtractedTriple] = Field(default_factory=list)
    changes_made: list[CorrectedItem] = Field(default_factory=list)
    issues_addressed: int = Field(default=0)


# ── Orphan Linker (post-verification orphan rescue) ────────────────

class OrphanClassification(BaseModel):
    """Classification of a single orphan entity."""
    entity_id: str = Field(description="The orphan entity text/id")
    action: str = Field(
        description="link (create new triples), reify (attach as property), or prune (remove)",
    )
    reason: str = Field(default="", description="Brief reason for this classification")
    target_entity: str = Field(
        default="",
        description="For reify: which existing entity this should be a property of",
    )
    property_name: str = Field(
        default="",
        description="For reify: the property/relation name (e.g. OCCURRED_ON, HAS_VALUE)",
    )


class OrphanNewTriple(BaseModel):
    """A new triple generated to link an orphan entity."""
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation type in UPPER_SNAKE_CASE")
    object: str = Field(description="Object entity")
    confidence: float = Field(default=0.7)
    evidence: str = Field(default="", description="Supporting text from the source")


class OrphanLinkerResponse(BaseModel):
    """Response from the orphan linker agent."""
    classifications: list[OrphanClassification] = Field(default_factory=list)
    new_triples: list[OrphanNewTriple] = Field(default_factory=list)
