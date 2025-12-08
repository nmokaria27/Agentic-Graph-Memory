"""
Ingestion agent for loading documents and raw text.
"""

from typing import Any, Optional
from pathlib import Path
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph


class IngestionAgent(Agent):
    """
    Agent responsible for ingesting documents from various sources.

    Supports:
    - File paths (text files)
    - Raw text strings
    """

    def __init__(
        self,
        name: str = "IngestionAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        """
        Initialize the ingestion agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to knowledge graph
        """
        super().__init__(name, knowledge_graph)

    def run(
        self,
        source: Optional[str] = None,
        text: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        """
        Ingest a document from a file or raw text.

        Args:
            source: File path to load
            text: Raw text content (if no source provided)
            encoding: File encoding (default: utf-8)
            **kwargs: Additional arguments

        Returns:
            Raw text content as a string

        Raises:
            ValueError: If neither source nor text is provided
            FileNotFoundError: If source file doesn't exist
        """
        if text is not None:
            self.log(f"Ingesting raw text ({len(text)} characters)")
            return text

        if source is not None:
            return self._load_from_file(source, encoding)

        raise ValueError("Either 'source' or 'text' must be provided")

    def _load_from_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Load text from a file.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            File contents as a string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.log(f"Loading document from: {file_path}")

        with open(path, "r", encoding=encoding) as f:
            content = f.read()

        self.log(f"Loaded {len(content)} characters from {path.name}")
        return content
