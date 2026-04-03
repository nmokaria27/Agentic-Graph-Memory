"""
Critic Agent (Phase 5 -- Critic-Corrector Verification Loop).

Implements the "critic" half of a critic-corrector reflection loop inspired by
FinReflectKG research.  The critic reviews extracted entities and triples
against the original source text and produces structured, item-level feedback
identifying:
    1. Factual errors
    2. Missing information
    3. Hallucinated content
    4. Type mismatches
    5. Redundancies

It also performs KARMA-style entity-relation consistency checks to ensure
every entity referenced in a triple actually appears in the entity list and
that entity types are used consistently.

The output is a ``CriticFeedback`` object obtained via constrained decoding
so downstream consumers always receive well-formed, parseable feedback.
"""

from typing import Any, Dict, List, Optional
import json

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
from multi_agent_kg.schemas.extraction_schemas import CriticFeedback


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph quality auditor. You will be given a "
    "source text together with entities and triples that were extracted from "
    "it. Your task is to critically review every extraction and identify "
    "specific, actionable issues. Be precise -- cite the problematic item "
    "and describe exactly what is wrong."
)

CRITIC_PROMPT = """\
Review the following extracted entities and triples against the source text.
For EACH item, determine whether it contains any of the following issues:

1. **Factual error** -- the extraction contradicts the source text.
2. **Missing info** -- the extraction omits important qualifiers or context
   that change its meaning.
3. **Hallucination** -- the extraction asserts something not stated or
   reasonably inferable from the source text.
4. **Type mismatch** -- an entity is assigned the wrong type, or a relation
   connects entity types that are semantically incompatible.
5. **Redundancy** -- two or more extractions capture the same fact.

SOURCE TEXT:
{source_text}

ENTITIES:
{entities_json}

TRIPLES:
{triples_json}

CONSISTENCY NOTES (auto-generated):
{consistency_notes}

Produce your review.  For every issue found, include the item_type
("entity" or "triple"), the problematic item_text, the issue_type (one of:
factual_error, missing_info, hallucination, type_mismatch, redundancy),
a clear description, a severity (low / medium / high), and a suggested_fix.

Also provide:
- entities_ok: true if entities pass review overall, false otherwise
- triples_ok: true if triples pass review overall, false otherwise
- overall_quality: a float between 0 and 1 summarising extraction quality
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CriticAgent(BaseAgent):
    """
    Critic Agent -- reviews extracted entities and triples for quality issues.

    Uses the deepseek-r1 reasoning model (ModelTier.LARGE) for thorough,
    chain-of-thought analysis.  Output is constrained to the
    ``CriticFeedback`` Pydantic schema via Ollama GBNF grammars.

    Responsibilities
    ----------------
    * Review entities and triples against source text.
    * Run entity-relation consistency checks (KARMA-style).
    * Produce structured ``CriticFeedback`` with per-item issues.

    Integration
    -----------
    * Reads from SharedMemory when prior context is useful.
    * Stores feedback in working memory so the CorrectorAgent (and any
      downstream agent) can retrieve it.
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        quality_threshold: float = 0.85,
    ):
        super().__init__(
            name="CriticAgent",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.LARGE,  # deepseek-r1 reasoning model
            quality_threshold=quality_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        context: AgentContext,
        entities: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Review extracted entities and triples, returning structured feedback.

        Parameters
        ----------
        context : AgentContext
            Processing context containing at minimum the source text.
        entities : list[dict], optional
            Extracted entities.  Falls back to ``context.entities``.
        triples : list[dict], optional
            Extracted triples.  Falls back to ``context.relations``.

        Returns
        -------
        ExtractionResult
            * ``items`` contains the full ``CriticFeedback`` dict.
            * ``confidence`` is the critic's ``overall_quality`` score.
            * ``metadata`` includes consistency-check details and counts.
        """
        self.stats["calls"] += 1

        entities = entities or context.entities or []
        triples = triples or context.relations or []

        # Fast path: nothing to review
        if not entities and not triples:
            empty_feedback = CriticFeedback(
                issues=[],
                entities_ok=True,
                triples_ok=True,
                overall_quality=1.0,
            )
            return ExtractionResult(
                items=empty_feedback.model_dump(),
                confidence=1.0,
                metadata={"status": "nothing_to_review"},
            )

        # Step 1: automated entity-relation consistency checks
        consistency_notes = self._check_entity_relation_consistency(
            entities, triples,
        )

        # Step 2: LLM-based critic review with constrained decoding
        feedback = self._run_critic_review(
            source_text=context.text,
            entities=entities,
            triples=triples,
            consistency_notes=consistency_notes,
        )

        # Step 3: persist feedback in working memory
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "critic_feedback": feedback,
                    "document_id": context.document_id,
                    "entity_count": len(entities),
                    "triple_count": len(triples),
                },
            )

        issue_count = len(feedback.get("issues", []))
        overall_quality = feedback.get("overall_quality", 0.0)
        self.log(
            f"Review complete: {issue_count} issues found, "
            f"overall_quality={overall_quality:.2f}"
        )

        return ExtractionResult(
            items=feedback,
            confidence=overall_quality,
            metadata={
                "document_id": context.document_id,
                "issue_count": issue_count,
                "entities_ok": feedback.get("entities_ok", True),
                "triples_ok": feedback.get("triples_ok", True),
                "consistency_issues": consistency_notes,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_entity_relation_consistency(
        self,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
    ) -> str:
        """
        KARMA-style cross-validation between entities and triples.

        Checks:
        * Every subject/object in triples exists in the entity list.
        * Entity types are used consistently across mentions.

        Returns a human-readable string of notes to inject into the LLM
        prompt so the critic is aware of structural problems.
        """
        if not entities or not triples:
            return "No consistency issues detected (insufficient data)."

        # Build lookup: normalised entity text -> list of assigned types
        entity_texts: Dict[str, List[str]] = {}
        for ent in entities:
            text = (ent.get("text") or ent.get("name") or "").strip().lower()
            etype = ent.get("type", "UNKNOWN")
            if text:
                entity_texts.setdefault(text, []).append(etype)

        notes: List[str] = []

        # --- Check 1: dangling references in triples ---
        for idx, triple in enumerate(triples):
            subj = (triple.get("subject") or "").strip().lower()
            obj = (triple.get("object") or "").strip().lower()

            if subj and subj not in entity_texts:
                notes.append(
                    f"Triple #{idx+1}: subject '{triple.get('subject')}' "
                    f"not found in entity list."
                )
            if obj and obj not in entity_texts:
                notes.append(
                    f"Triple #{idx+1}: object '{triple.get('object')}' "
                    f"not found in entity list."
                )

        # --- Check 2: inconsistent entity types ---
        for text, types in entity_texts.items():
            unique_types = set(types)
            if len(unique_types) > 1:
                notes.append(
                    f"Entity '{text}' has inconsistent types: "
                    f"{', '.join(sorted(unique_types))}."
                )

        if not notes:
            return "No structural consistency issues detected."

        return "\n".join(notes)

    def _run_critic_review(
        self,
        source_text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        consistency_notes: str,
    ) -> Dict[str, Any]:
        """
        Call the LLM with constrained decoding to produce ``CriticFeedback``.

        Long texts/entity lists are truncated to stay within token limits.
        """
        entities_json = json.dumps(entities[:80], indent=2)
        triples_json = json.dumps(triples[:80], indent=2)

        prompt = CRITIC_PROMPT.format(
            source_text=source_text[:6000],
            entities_json=entities_json,
            triples_json=triples_json,
            consistency_notes=consistency_notes,
        )

        try:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                tier=ModelTier.LARGE,
                max_tokens=4096,
                response_schema=CriticFeedback,
            )
        except Exception as exc:
            self.log(f"LLM call failed, returning default feedback: {exc}", level="WARNING")
            result = CriticFeedback(
                issues=[],
                entities_ok=True,
                triples_ok=True,
                overall_quality=0.5,
            ).model_dump()

        # Normalise: if the LLM returned a Pydantic model, convert to dict
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return result
