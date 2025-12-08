"""
OpenAI client wrapper for LLM interactions.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Call OpenAI chat.completions and return the assistant message content as a string.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The OpenAI model to use (default: gpt-4)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        The assistant's response as a string

    Raises:
        Exception: If the API call fails
    """
    try:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        if content is None:
            return ""
        
        return content

    except Exception as e:
        raise Exception(f"OpenAI API call failed: {str(e)}")


def chat_completion_json(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Call OpenAI chat.completions requesting JSON output and parse the response.

    This function instructs the model to return valid JSON and parses it.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The OpenAI model to use (default: gpt-4)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        Parsed JSON object (dict, list, etc.)

    Raises:
        json.JSONDecodeError: If the response is not valid JSON
        Exception: If the API call fails
    """
    # Modify the last user message to explicitly request JSON
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        if "JSON" not in original_content and "json" not in original_content:
            modified_messages[-1]["content"] = (
                f"{original_content}\n\nReturn your response as valid JSON."
            )

    # Get the response
    response_text = chat_completion(
        messages=modified_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Try to parse JSON
    try:
        # Sometimes models wrap JSON in markdown code blocks
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        # If parsing fails, try to extract JSON from the response
        try:
            # Look for JSON object or array
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start == -1:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
            
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                return json.loads(json_str)
        except:
            pass

        raise json.JSONDecodeError(
            f"Failed to parse JSON from response: {response_text[:200]}...",
            response_text,
            0,
        )


def get_embedding(
    text: str,
    model: str = "text-embedding-ada-002",
) -> List[float]:
    """
    Get an embedding vector for the given text.

    Args:
        text: The text to embed
        model: The embedding model to use

    Returns:
        List of floats representing the embedding vector

    Raises:
        Exception: If the API call fails
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    except Exception as e:
        raise Exception(f"OpenAI embedding API call failed: {str(e)}")
