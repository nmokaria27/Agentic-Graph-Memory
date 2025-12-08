"""
Base agent class for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.messages import Message, MessageType


class Agent(ABC):
    """
    Base class for all agents in the multi-agent system.

    All agents inherit from this class and implement the `run` method.
    """

    def __init__(self, name: str, knowledge_graph: Optional[KnowledgeGraph] = None):
        """
        Initialize the agent.

        Args:
            name: Name of the agent
            knowledge_graph: Optional reference to the shared knowledge graph
        """
        self.name = name
        self.knowledge_graph = knowledge_graph

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the agent's main task.

        This method should be implemented by all subclasses.

        Args:
            **kwargs: Arbitrary keyword arguments specific to the agent

        Returns:
            Result of the agent's processing (type varies by agent)
        """
        pass

    def log(self, msg: str, level: str = "INFO") -> None:
        """
        Log a message from this agent.

        Args:
            msg: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        print(f"[{level}] [{self.name}] {msg}")

    def create_message(
        self,
        receiver: str,
        message_type: MessageType,
        payload: dict,
        meta: Optional[dict] = None,
    ) -> Message:
        """
        Create a message from this agent.

        Args:
            receiver: Name of the receiving agent
            message_type: Type of message
            payload: Message payload
            meta: Optional metadata

        Returns:
            Message object
        """
        return Message(
            sender=self.name,
            receiver=receiver,
            type=message_type,
            payload=payload,
            meta=meta or {},
        )
