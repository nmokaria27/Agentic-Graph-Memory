"""
Corrector Agent (Phase 5 -- Critic-Corrector Verification Loop).

Implements the "corrector" half of a critic-corrector reflection loop inspired
by FinReflectKG research.  The corrector receives extracted entities, triples,
and structured critic feedback, then produces a corrected set of entities and
triples that addresses every identified issue.

For each issue the corrector either:
    * Fixes the problematic extraction.
    * Removes it entirely (if the issue is unfixable, e.g. hallucination).
    * Keeps it unchanged with an explanation of why the critic's concern is
      unwarranted.

The output is a ``CorrectorResponse`` obtained via constrained decoding so
downstream consumers always receive well-formed, parseable corrections.
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
from multi_agent_kg.schemas.extraction_schemas import CorrectorResponse


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CORRECTOR_SYSTEM_PROMPT = (
    "You are an expert knowledge-graph corrector. You receive extracted "
    "entities and triples together with detailed critic feedback listing "
    "specific issues. Your job is to produce a corrected set of entities "
    "and triples that resolves every issue. Be precise and conservative: "
    "only change what is necessary."
)

CORRECTOR_PROMPT = """\
You are given extracted entities and triples, the original source text,
and a list of issues identified by a critic agent.

For EACH issue, do one of the following:
- **fix**: Correct the entity or triple to resolve the problem.
- **remove**: Delete the extraction if it is hallucinated or unfixable.
- **keep**: Retain the extraction unchanged and explain why the critic's
  concern does not apply.

SOURCE TEXT:
{source_text}

CURRENT ENTITIES:
{entities_json}

CURRENT TRIPLES:
{triples_json}

CRITIC FEEDBACK (issues to address):
{feedback_json}

Produce:
- corrected_entities: the full corrected entity list (including unchanged ones).
- corrected_triples: the full corrected triple list (including unchanged ones).
- changes_made: for each change, the original text, corrected text, action
  (fix / remove / keep), and a brief reason.
- issues_addressed: the total number of critic issues you addressed.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CorrectorAgent(BaseAgent):
    """
    Corrector Agent -- fixes issues identified by the CriticAgent.

    Uses a medium-tier model (ModelTier.MEDIUM) since the reasoning-heavy
    work was already done by the critic; the corrector primarily applies
    targeted edits.

    Responsibilities
    ----------------
    * Parse critic feedback and map issues to specific entities/triples.
    * Produce corrected entities and triples via constrained decoding.
    * Record the changelog (what was fixed, removed, or kept).

    Integration
    -----------
    * Reads critic feedback from working memory (or directly from args).
    * Stores the corrected output in working memory for downstream agents.
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
            name="CorrectorAgent",
            role=AgentRole.COORDINATOR,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.MEDIUM,
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
        critic_feedback: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Correct entities and triples based on critic feedback.

        Parameters
        ----------
        context : AgentContext
            Processing context containing at minimum the source text.
        entities : list[dict], optional
            Current entities.  Falls back to ``context.entities``.
        triples : list[dict], optional
            Current triples.  Falls back to ``context.relations``.
        critic_feedback : dict, optional
            Structured feedback from ``CriticAgent``.  If not provided,
            the agent attempts to retrieve it from shared memory.

        Returns
        -------
        ExtractionResult
            * ``items`` contains ``corrected_entities`` and
              ``corrected_triples`` plus a ``changes_made`` changelog.
            * ``confidence`` reflects the ratio of issues addressed.
            * ``metadata`` includes correction statistics.
        """
        self.stats["calls"] += 1

        entities = entities or context.entities or []
        triples = triples or context.relations or []

        # Attempt to retrieve critic feedback from memory if not provided
        if critic_feedback is None:
            critic_feedback = self._retrieve_critic_feedback()

        # If there is still no feedback, or no issues to fix, pass through
        if not critic_feedback or not critic_feedback.get("issues"):
            self.log("No critic feedback or no issues -- returning inputs unchanged.")
            return ExtractionResult(
                items={
                    "corrected_entities": entities,
                    "corrected_triples": triples,
                    "changes_made": [],
                },
                confidence=1.0,
                metadata={"status": "no_issues_to_correct", "issues_addressed": 0},
            )

        # Nothing to correct if both lists are empty
        if not entities and not triples:
            return ExtractionResult(
                items={
                    "corrected_entities": [],
                    "corrected_triples": [],
                    "changes_made": [],
                },
                confidence=1.0,
                metadata={"status": "nothing_to_correct"},
            )

        # Run the corrector LLM call
        correction_result = self._run_correction(
            source_text=context.text,
            entities=entities,
            triples=triples,
            critic_feedback=critic_feedback,
        )

        corrected_entities = correction_result.get("corrected_entities", entities)
        corrected_triples = correction_result.get("corrected_triples", triples)
        changes_made = correction_result.get("changes_made", [])
        issues_addressed = correction_result.get("issues_addressed", 0)
        total_issues = len(critic_feedback.get("issues", []))

        # Persist results in working memory
        if self.shared_memory:
            self.store_in_memory(
                memory_type=MemoryType.WORKING,
                content={
                    "corrected_entities": corrected_entities,
                    "corrected_triples": corrected_triples,
                    "changes_made": changes_made,
                    "issues_addressed": issues_addressed,
                    "document_id": context.document_id,
                },
            )

        confidence = (
            issues_addressed / total_issues if total_issues > 0 else 1.0
        )

        self.log(
            f"Correction complete: {issues_addressed}/{total_issues} issues addressed, "
            f"{len(changes_made)} changes made"
        )

        return ExtractionResult(
            items={
                "corrected_entities": corrected_entities,
                "corrected_triples": corrected_triples,
                "changes_made": changes_made,
            },
            confidence=confidence,
            metadata={
                "document_id": context.document_id,
                "issues_addressed": issues_addressed,
                "total_issues": total_issues,
                "changes_count": len(changes_made),
                "entity_count": len(corrected_entities),
                "triple_count": len(corrected_triples),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_critic_feedback(self) -> Optional[Dict[str, Any]]:
        """
        Try to retrieve the most recent critic feedback from shared memory.

        Returns ``None`` if shared memory is unavailable or no feedback is
        found.
        """
        if not self.shared_memory:
            return None

        entries = self.retrieve_from_memory(
            memory_type=MemoryType.WORKING,
            limit=5,
        )
        for entry in entries:
            content = getattr(entry, "content", entry) if not isinstance(entry, dict) else entry
            if isinstance(content, dict) and "critic_feedback" in content:
                return content["critic_feedback"]

        return None

    def _run_correction(
        self,
        source_text: str,
        entities: List[Dict[str, Any]],
        triples: List[Dict[str, Any]],
        critic_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call the LLM with constrained decoding to produce ``CorrectorResponse``.

        Long inputs are truncated to stay within token limits.
        """
        entities_json = json.dumps(entities[:80], indent=2)
        triples_json = json.dumps(triples[:80], indent=2)

        # Serialise critic feedback; handle both dict and Pydantic model
        if hasattr(critic_feedback, "model_dump"):
            feedback_dict = critic_feedback.model_dump()
        else:
            feedback_dict = critic_feedback
        feedback_json = json.dumps(feedback_dict, indent=2)

        prompt = CORRECTOR_PROMPT.format(
            source_text=source_text[:6000],
            entities_json=entities_json,
            triples_json=triples_json,
            feedback_json=feedback_json,
        )

        try:
            result = self.call_llm(
                prompt=prompt,
                system_prompt=CORRECTOR_SYSTEM_PROMPT,
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
                response_schema=CorrectorResponse,
            )
        except Exception as exc:
            self.log(
                f"LLM call failed, returning inputs unchanged: {exc}",
                level="WARNING",
            )
            result = CorrectorResponse(
                corrected_entities=[],
                corrected_triples=[],
                changes_made=[],
                issues_addressed=0,
            ).model_dump()

        # Normalise: if the LLM returned a Pydantic model, convert to dict
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        # Fallback: if the corrector returned empty lists but we had inputs,
        # preserve the originals (the LLM may have omitted unchanged items).
        if not result.get("corrected_entities") and entities:
            result["corrected_entities"] = entities
        if not result.get("corrected_triples") and triples:
            result["corrected_triples"] = triples

        return result
