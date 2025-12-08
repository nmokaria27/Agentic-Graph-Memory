"""LLM module initialization."""

from multi_agent_kg.llm.openai_client import chat_completion, chat_completion_json

__all__ = ["chat_completion", "chat_completion_json"]
