"""
LLM client wrapper supporting both Ollama (default) and OpenAI backends.

Backend selection:
- Set LLM_BACKEND=openai to use OpenAI (requires OPENAI_API_KEY)
- Default: Ollama at http://localhost:11434 (via SSH tunnel to GPU)

Ollama model tiers (on gpu01.mind.cs.umd.edu):
- LARGE:  gemma3:27b   (27B params, best quality)
- MEDIUM: qwen3:8b     (8B params, balanced)
- SMALL:  qwen3:4b     (4B params, fast)
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Backend configuration ─────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if LLM_BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    # Ollama's OpenAI-compatible endpoint; no API key needed
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


# ── Default Ollama model mapping ──────────────────────────────────────
OLLAMA_MODELS = {
    "large": "gemma3:27b",
    "medium": "gemma3:27b",
    "small": "gemma3:27b",
}

# Map old OpenAI model names → Ollama equivalents (for backward compat)
_OPENAI_TO_OLLAMA = {
    "gpt-4o": "gemma3:27b",
    "gpt-4o-mini": "gemma3:27b",
    "gpt-4": "gemma3:27b",
    "gpt-4-turbo": "gemma3:27b",
    "gpt-3.5-turbo": "gemma3:27b",
}


def _resolve_model(model: str) -> str:
    """Resolve model name: translate OpenAI names to Ollama when using Ollama backend."""
    if LLM_BACKEND == "openai":
        return model
    # If the caller passed an OpenAI model name, translate it
    return _OPENAI_TO_OLLAMA.get(model, model)


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gemma3:27b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Call LLM chat completions and return the assistant message content as a string.

    Works with both Ollama and OpenAI backends transparently.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name (Ollama or OpenAI names both accepted)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        The assistant's response as a string
    """
    resolved_model = _resolve_model(model)

    # Filter out kwargs that Ollama doesn't support
    filtered_kwargs = {}
    if LLM_BACKEND == "openai":
        filtered_kwargs = kwargs
    # Ollama silently ignores unknown params via the OpenAI-compat layer

    try:
        params: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            **filtered_kwargs,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content

        return content if content is not None else ""

    except Exception as e:
        err = str(e)
        # For OpenAI backend, try quota fallback chain
        if LLM_BACKEND == "openai" and "insufficient_quota" in err:
            _QUOTA_FALLBACK = {
                "gpt-4o": "gpt-4o-mini",
                "gpt-4o-mini": "gpt-3.5-turbo",
            }
            if resolved_model in _QUOTA_FALLBACK:
                fallback = _QUOTA_FALLBACK[resolved_model]
                print(f"  Quota exceeded for {resolved_model}, retrying with {fallback}")
                return chat_completion(
                    messages=messages,
                    model=fallback,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
        raise Exception(f"LLM API call failed ({resolved_model}): {err}")


def chat_completion_json(
    messages: List[Dict[str, str]],
    model: str = "gemma3:27b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Call LLM requesting JSON output and parse the response.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        Parsed JSON object (dict, list, etc.)
    """
    # Modify the last user message to explicitly request JSON
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        if "JSON" not in original_content and "json" not in original_content:
            modified_messages[-1]["content"] = (
                f"{original_content}\n\nReturn your response as valid JSON only, with no additional text."
            )

    # For Ollama models that support thinking (qwen3), disable thinking for JSON
    # by appending /no_think to avoid <think> blocks in output
    resolved_model = _resolve_model(model)

    # Get the response
    response_text = chat_completion(
        messages=modified_messages,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Strip any <think>...</think> blocks (qwen3 thinking mode)
    import re
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

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
        # If parsing fails, try to extract and fix JSON from the response
        try:
            # Look for JSON object or array
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start == -1:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1

            if start != -1 and end > start:
                json_str = cleaned[start:end]

                # Try multiple fix strategies
                fix_attempts = [
                    json_str,  # Original
                    json_str.replace(",]", "]").replace(",}", "}"),  # Trailing commas
                    re.sub(r',(\s*[}\]])', r'\1', json_str),  # Remove commas before closing brackets
                    re.sub(r'([}\]])(\s*)(["\'{[])', r'\1,\2\3', json_str),  # Add missing commas
                ]

                for attempt in fix_attempts:
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        continue

                # If still fails, might be truncated - try to close it
                json_str = fix_attempts[-1]
                if json_str.count("{") > json_str.count("}"):
                    json_str = json_str + "}" * (json_str.count("{") - json_str.count("}"))
                if json_str.count("[") > json_str.count("]"):
                    json_str = json_str + "]" * (json_str.count("[") - json_str.count("]"))

                try:
                    return json.loads(json_str)
                except Exception:
                    pass
        except Exception:
            pass

        # Last resort: return empty structure based on what was expected
        print(f"  WARNING: Failed to parse JSON, returning empty result. Error: {str(e)}")
        print(f"  Raw response (first 500 chars): {response_text[:500]}")
        if "linked_triples" in response_text or "triples" in response_text:
            return {"linked_triples": [], "triples": []}
        elif "entities" in response_text:
            return {"entities": []}
        elif "relations" in response_text or "relation" in response_text:
            return {"relations": [], "relations_found": []}
        else:
            return {}


def get_embedding(
    text: str,
    model: str = "text-embedding-ada-002",
) -> List[float]:
    """
    Get an embedding vector for the given text.

    Note: Ollama embedding support varies by model. Falls back to a
    simple hash-based placeholder if not available.
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    except Exception as e:
        if LLM_BACKEND != "openai":
            # Ollama may not support all embedding models; return placeholder
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            # Generate a deterministic 256-dim pseudo-embedding from the hash
            return [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), 512), 2)]
        raise Exception(f"Embedding API call failed: {str(e)}")
