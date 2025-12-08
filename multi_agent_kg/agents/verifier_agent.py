"""
Verification agent for scoring and approving triples based on evidence.
"""

from typing import Any, List, Optional
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion_json


class VerifierAgent(Agent):
    """
    Agent responsible for verifying triples against source text.

    Uses LLM to score each triple for:
    - Textual support
    - Plausibility
    - Confidence
    """

    def __init__(
        self,
        name: str = "VerifierAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the verification agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
            llm_config: Configuration for LLM calls
            confidence_threshold: Minimum confidence to approve a triple
        """
        super().__init__(name, knowledge_graph)
        self.llm_config = llm_config or LLMConfig(temperature=0.1)
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        triples: List[Triple],
        source_text: str,
        batch_size: int = 5,
        **kwargs: Any,
    ) -> List[Triple]:
        """
        Verify triples against source text.

        Args:
            triples: List of triples to verify
            source_text: Original source text
            batch_size: Number of triples to verify per LLM call
            **kwargs: Additional arguments

        Returns:
            List of approved triples
        """
        self.log(f"Verifying {len(triples)} triples")

        approved_triples = []

        # Process in batches
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            self.log(f"Verifying batch {i // batch_size + 1}")

            verified = self._verify_batch(batch, source_text)
            approved_triples.extend(verified)

        self.log(
            f"Verification complete: {len(approved_triples)}/{len(triples)} "
            f"triples approved (threshold: {self.confidence_threshold})"
        )

        return approved_triples

    def _verify_batch(self, triples: List[Triple], source_text: str) -> List[Triple]:
        """
        Verify a batch of triples.

        Args:
            triples: List of triples to verify
            source_text: Source text

        Returns:
            List of approved triples
        """
        # Format triples for prompt
        triple_strs = []
        for i, triple in enumerate(triples):
            triple_strs.append(
                f"{i + 1}. ({triple.subject}) -[{triple.relation}]-> ({triple.object})"
            )
        triples_text = "\n".join(triple_strs)

        system_msg = {
            "role": "system",
            "content": (
                "You are a fact verification system. "
                "Your task is to determine whether each given triple is supported by the text. "
                "Be strict: only approve triples that are clearly stated or strongly implied."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Source text:\n{source_text}\n\n"
                f"Triples to verify:\n{triples_text}\n\n"
                f"For each triple, determine:\n"
                f"- supported: true if the triple is supported by the text, false otherwise\n"
                f"- confidence: confidence score from 0.0 to 1.0\n"
                f"- rationale: brief explanation (1 sentence)\n\n"
                f"Return a JSON object with a 'verifications' key containing an array "
                f"where each element corresponds to a triple in order."
            ),
        }

        try:
            response = chat_completion_json(
                messages=[system_msg, user_msg],
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens or 1500,
            )

            verifications = response.get("verifications", [])
            approved = []

            for i, verification in enumerate(verifications):
                if i >= len(triples):
                    break

                triple = triples[i]
                supported = verification.get("supported", False)
                confidence = verification.get("confidence", 0.0)
                rationale = verification.get("rationale", "")

                if supported and confidence >= self.confidence_threshold:
                    # Update triple confidence
                    triple.confidence = confidence
                    if "verification" not in triple.metadata:
                        triple.metadata["verification"] = {}
                    triple.metadata["verification"]["rationale"] = rationale
                    approved.append(triple)
                    self.log(
                        f"✓ Approved: {triple} (confidence: {confidence:.2f})",
                    )
                else:
                    self.log(
                        f"✗ Rejected: {triple} "
                        f"(supported: {supported}, confidence: {confidence:.2f})",
                        level="WARNING",
                    )

            return approved

        except Exception as e:
            self.log(f"Verification failed: {e}", level="ERROR")
            # In case of error, return triples with existing confidence
            return [t for t in triples if (t.confidence or 0.0) >= self.confidence_threshold]
