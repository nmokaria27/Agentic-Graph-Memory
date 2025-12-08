"""
Summarizer agent for generating summaries of text segments.
"""

from typing import Any, List, Optional
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.llm.openai_client import chat_completion


class SummarizerAgent(Agent):
    """
    Agent responsible for summarizing text segments using LLM.
    """

    def __init__(
        self,
        name: str = "SummarizerAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the summarizer agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
            llm_config: Configuration for LLM calls
        """
        super().__init__(name, knowledge_graph)
        self.llm_config = llm_config or LLMConfig(temperature=0.3)

    def run(
        self,
        segments: List[str],
        max_summary_sentences: int = 3,
        **kwargs: Any,
    ) -> List[str]:
        """
        Summarize a list of text segments.

        Args:
            segments: List of text segments to summarize
            max_summary_sentences: Maximum sentences per summary
            **kwargs: Additional arguments

        Returns:
            List of summaries (one per segment)
        """
        self.log(f"Summarizing {len(segments)} segments")

        summaries = []
        for i, segment in enumerate(segments):
            self.log(f"Summarizing segment {i + 1}/{len(segments)}")
            summary = self._summarize_segment(segment, max_summary_sentences)
            summaries.append(summary)

        self.log(f"Generated {len(summaries)} summaries")
        return summaries

    def _summarize_segment(self, text: str, max_sentences: int) -> str:
        """
        Summarize a single text segment.

        Args:
            text: Text to summarize
            max_sentences: Maximum sentences in summary

        Returns:
            Summary as a string
        """
        # If text is already short, return as-is
        if len(text) < 200:
            return text

        system_msg = {
            "role": "system",
            "content": (
                "You are a precise summarization assistant. "
                "Summarize the given text concisely while preserving key facts and entities."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Summarize the following text in {max_sentences} sentences or fewer. "
                f"Focus on key facts, entities, and relationships.\n\n"
                f"Text:\n{text}"
            ),
        }

        try:
            summary = chat_completion(
                messages=[system_msg, user_msg],
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            return summary.strip()

        except Exception as e:
            self.log(f"Summarization failed: {e}", level="ERROR")
            # Fallback: return truncated original text
            return text[:500] + "..."
