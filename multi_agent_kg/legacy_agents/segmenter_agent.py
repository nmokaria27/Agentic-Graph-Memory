"""
Segmenter agent for splitting documents into chunks.
"""

from typing import Any, List, Optional
import re
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


class SegmenterAgent(Agent):
    """
    Agent responsible for segmenting documents into smaller chunks.

    Supports various segmentation strategies:
    - Paragraph-based
    - Sentence-based
    - Fixed-size chunks
    """

    def __init__(
        self,
        name: str = "SegmenterAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
        min_segment_length: int = 50,
        strategy: str = "paragraph",
    ):
        """
        Initialize the segmenter agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
            min_segment_length: Minimum character length for segments
            strategy: Segmentation strategy ('paragraph', 'sentence', 'fixed')
        """
        super().__init__(name, knowledge_graph)
        self.min_segment_length = min_segment_length
        self.strategy = strategy

    def run(
        self,
        text: str,
        strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Segment text into chunks.

        Args:
            text: Input text to segment
            strategy: Override default segmentation strategy
            **kwargs: Additional arguments

        Returns:
            List of text segments
        """
        strategy = strategy or self.strategy

        self.log(f"Segmenting text using '{strategy}' strategy")

        if strategy == "paragraph":
            segments = self._segment_by_paragraph(text)
        elif strategy == "sentence":
            segments = self._segment_by_sentence(text)
        elif strategy == "fixed":
            chunk_size = kwargs.get("chunk_size", 1000)
            segments = self._segment_fixed_size(text, chunk_size)
        else:
            self.log(f"Unknown strategy '{strategy}', using paragraph", level="WARNING")
            segments = self._segment_by_paragraph(text)

        # Filter out very short segments
        segments = [s for s in segments if len(s) >= self.min_segment_length]

        self.log(f"Created {len(segments)} segments")
        return segments

    def _segment_by_paragraph(self, text: str) -> List[str]:
        """
        Segment text by paragraphs (double newline).

        Args:
            text: Input text

        Returns:
            List of paragraph segments
        """
        # Split on double newlines or more
        segments = re.split(r"\n\s*\n", text)
        # Clean up whitespace
        segments = [s.strip() for s in segments if s.strip()]
        return segments

    def _segment_by_sentence(self, text: str) -> List[str]:
        """
        Segment text by sentences.

        Args:
            text: Input text

        Returns:
            List of sentence segments
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _segment_fixed_size(self, text: str, chunk_size: int) -> List[str]:
        """
        Segment text into fixed-size chunks.

        Args:
            text: Input text
            chunk_size: Size of each chunk in characters

        Returns:
            List of fixed-size segments
        """
        segments = []
        for i in range(0, len(text), chunk_size):
            segment = text[i : i + chunk_size].strip()
            if segment:
                segments.append(segment)
        return segments
