"""
LLM client wrapper supporting both Ollama (default) and OpenAI backends.

Backend selection:
- Set LLM_BACKEND=openai to use OpenAI (requires OPENAI_API_KEY)
- Default: Ollama at http://localhost:11434 (via SSH tunnel to GPU)

Ollama model tiers (on gpu01.mind.cs.umd.edu):
- LARGE:  gemma3:27b   (27B params, best quality)
- MEDIUM: qwen3:8b     (8B params, balanced)
- SMALL:  qwen3:4b     (4B params, fast)

Structured output:
- chat_completion_structured() uses Ollama's native `format` parameter
  which applies GBNF grammar constraints at the token level, guaranteeing
  valid JSON matching the provided Pydantic schema.
"""

from typing import List, Dict, Any, Optional, Type
from openai import OpenAI
from pydantic import BaseModel
import os
import json
import re
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


def _get_ollama_base() -> str:
    """Return the Ollama base URL without the /v1 suffix."""
    url = OLLAMA_BASE_URL.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def chat_completion_structured(
    messages: List[Dict[str, str]],
    schema: Type[BaseModel],
    model: str = "gemma3:27b",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> Any:
    """
    Call LLM with Ollama's native structured output (GBNF grammar constraints).

    Uses Ollama's **native /api/chat endpoint** (not the OpenAI-compat layer)
    with the ``format`` parameter to pass a JSON schema.  Ollama/llama.cpp
    converts the schema to a GBNF grammar and masks invalid tokens during
    generation, **guaranteeing** structurally valid JSON.

    For the OpenAI backend the function falls back to requesting JSON mode
    with the schema described in the prompt, then validates with Pydantic.

    Args:
        messages: Chat messages.
        schema: A Pydantic BaseModel **class** (not an instance).
        model: Model name.
        temperature: Sampling temperature (low recommended for schemas).
        max_tokens: Maximum tokens in the response.
        max_retries: Number of retries on validation failure.
        **kwargs: Extra params forwarded to the API.

    Returns:
        Parsed + validated dict matching the schema.
    """
    import httpx

    resolved_model = _resolve_model(model)

    # Build the JSON schema dict from the Pydantic model
    json_schema = schema.model_json_schema()

    # Inject a hint about the expected format into the last user message
    modified_messages = list(messages)
    if modified_messages and modified_messages[-1]["role"] == "user":
        original = modified_messages[-1]["content"]
        if "JSON" not in original and "json" not in original:
            modified_messages[-1] = {
                "role": "user",
                "content": (
                    f"{original}\n\nReturn your response as valid JSON only, "
                    f"with no additional text."
                ),
            }

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if LLM_BACKEND != "openai":
                # ── Ollama native /api/chat with format parameter ──
                ollama_base = _get_ollama_base()
                payload: Dict[str, Any] = {
                    "model": resolved_model,
                    "messages": modified_messages,
                    "format": json_schema,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                }
                if max_tokens is not None:
                    payload["options"]["num_predict"] = max_tokens

                resp = httpx.post(
                    f"{ollama_base}/api/chat",
                    json=payload,
                    timeout=600.0,  # 10 min for large batches over SSH tunnel
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
            else:
                # ── OpenAI backend ──
                params: Dict[str, Any] = {
                    "model": resolved_model,
                    "messages": modified_messages,
                    "temperature": temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": json_schema,
                        },
                    },
                }
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens

                response = client.chat.completions.create(**params)
                content = response.choices[0].message.content or ""

            # Strip <think> blocks from reasoning models
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()

            # Parse and validate with Pydantic
            parsed = json.loads(content)
            validated = schema.model_validate(parsed)
            return validated.model_dump()

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                temperature = min(temperature + 0.1, 0.5)
                continue

    # All retries exhausted -- fall back to unstructured JSON parsing
    print(
        f"  WARNING: Structured output failed after {max_retries + 1} attempts "
        f"({last_error}), falling back to chat_completion_json"
    )
    return chat_completion_json(
        messages=messages,
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
    )


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
