"""
Pydantic v2 schemas for validating LLM JSON output at the agent boundary.

These models sit *after* the robust text->dict parser in
``multi_agent_kg/llm/openai_client.py`` (``_extract_json``). They do not parse
messy LLM text; they validate the already-parsed dict, coercing the salvageable
and dropping only truly-broken items so a malformed response degrades gracefully
instead of crashing the pipeline.

Policy: lenient. Field names mirror the exact prompt schemas in the agents so no
prompt edits are required and downstream dict access stays valid.
"""

from typing import Any, Type, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_Item = TypeVar("_Item", bound="_LLMModel")


def _filter_valid_items(raw: Any, item_model: Type[_Item], field: str) -> Any:
    """Drop individually-broken items from a list field instead of failing the
    whole response. Each element is validated against ``item_model``; elements
    that raise ``ValidationError`` (or aren't dicts) are skipped and logged.

    Returns the cleaned dict so the outer model only ever sees valid items.
    A bare top-level list is treated as the items list (LLMs sometimes drop the
    wrapper key). A bare scalar (e.g. a thinking model's raw CoT prose — the
    original line-611 crash input) degrades to an empty result so even a direct
    ``model_validate`` is crash-proof. A dict whose field is the wrong type is
    left untouched so the outer model raises a clear error the caller can turn
    into a re-prompt.
    """
    if isinstance(raw, list):
        raw = {field: raw}
    if not isinstance(raw, dict):
        return {field: []}
    items = raw.get(field)
    if not isinstance(items, list):
        return raw
    kept: list[Any] = []
    dropped = 0
    for it in items:
        try:
            kept.append(item_model.model_validate(it))
        except ValidationError:
            dropped += 1
    if dropped:
        print(f"  WARNING: dropped {dropped} malformed item(s) from '{field}'")
    cleaned = dict(raw)
    cleaned[field] = kept
    return cleaned


class _LLMModel(BaseModel):
    """Base for all LLM-output models.

    ``extra="ignore"`` tolerates stray keys the model invents (don't reject a
    whole item over an unexpected field). ``str_strip_whitespace`` trims the
    whitespace LLMs love to add around values.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


def _coerce_confidence(v: object) -> float:
    """Coerce a confidence value to a float clamped to [0, 1].

    Handles the common LLM quirks: confidence emitted as a string ("0.8"),
    as null, or out of range. Falls back to 0.5 when uncoercible.
    """
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5


class EntityOut(_LLMModel):
    """One extracted entity. Mirrors the entity-extraction prompt schema:
    ``{"text": ..., "type": ..., "confidence": ...}``.
    """

    text: str
    type: str = "UNKNOWN"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: object) -> float:
        return _coerce_confidence(v)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entity text empty")
        return v


class EntityExtractionOut(_LLMModel):
    """Top-level entity-extraction response: ``{"entities": [EntityOut, ...]}``."""

    entities: list[EntityOut] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_bad(cls, data: Any) -> Any:
        return _filter_valid_items(data, EntityOut, "entities")


class EntityGroupOut(_LLMModel):
    """One coreference group. Mirrors the COREFERENCE_PROMPT schema:
    ``{"canonical_id", "canonical_name", "type", "mentions", "is_known_entity"}``.
    """

    canonical_id: str = ""
    canonical_name: str = ""
    type: str = "UNKNOWN"
    mentions: list[str] = Field(default_factory=list)
    is_known_entity: bool = False


class CoreferenceOut(_LLMModel):
    """Top-level coreference response: ``{"entity_groups": [EntityGroupOut, ...]}``."""

    entity_groups: list[EntityGroupOut] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_bad(cls, data: Any) -> Any:
        return _filter_valid_items(data, EntityGroupOut, "entity_groups")


# ── Relation-extractor item models ────────────────────────────────────
# These validate the per-item dicts that ``relation_extractor._coerce_llm_items``
# already flattens out of the LLM response. Goal: add structure + coercion
# (confidence -> clamped float, pair_index -> int-or-None) WITHOUT dropping any
# item the pipeline would previously have kept. To guarantee no yield loss they:
#   * make every known field optional with a safe default, and
#   * use ``extra="allow"`` so any other key the LLM/pipeline emitted (e.g. the
#     ``relation_type`` / ``predicate`` aliases read downstream) survives
#     ``model_dump`` untouched.
# With all fields optional + extra allowed, ``model_validate`` on a dict can
# never raise, so validation is pure normalization — never a filter.


def _coerce_int_or_none(v: object) -> object:
    """Coerce to int when possible; otherwise leave as None. Never raises, so a
    junk ``pair_index`` degrades to 'no match' rather than dropping the item."""
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class _RelationItemModel(BaseModel):
    """Base for relation/triple items: lossless (keeps unknown keys) and trims."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class PredictionOut(_RelationItemModel):
    """One pairwise relation prediction. Mirrors the SciERC pairwise prompt:
    ``{"pair_index", "relation", "confidence", "evidence", "direction_rationale"}``.
    ``relation`` is left as-is (incl. empty / "NONE"); downstream applies the
    schema/NONE filtering so yield decisions stay with the business logic.
    """

    pair_index: Any = None
    relation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""
    direction_rationale: str = ""

    @field_validator("pair_index", mode="before")
    @classmethod
    def _pi(cls, v: object) -> object:
        return _coerce_int_or_none(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: object) -> float:
        return _coerce_confidence(v)


class TripleOut(_RelationItemModel):
    """One extracted triple. Mirrors the head/tail-binding prompt shape; all
    fields optional so partial triples survive to the existing downstream
    guards (which require subject+object before committing).
    """

    subject: str = ""
    subject_id: str = ""
    relation: str = ""
    object: str = ""
    object_id: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: object) -> float:
        return _coerce_confidence(v)
