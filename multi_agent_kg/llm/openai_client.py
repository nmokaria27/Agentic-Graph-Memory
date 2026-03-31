"""
LLM client wrapper supporting both Ollama (default) and OpenAI backends.

Backend selection:
- Set LLM_BACKEND=openai to use OpenAI (requires OPENAI_API_KEY)
- Default: Ollama at http://localhost:11434 (via SSH tunnel to GPU)
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

# ── Backend configuration ─────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Timeout for Ollama calls (local models can be slow)
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # 5 min default

if LLM_BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=_TIMEOUT,
    )

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
    return _OPENAI_TO_OLLAMA.get(model, model)


def _extract_json(text: str) -> Any:
    """
    Robustly extract and parse JSON from LLM output that may contain
    markdown fences, preamble text, thinking tags, or trailing commentary.
    """
    if not text or not text.strip():
        return None

    # 1. Strip <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip markdown code fences
    cleaned = text.strip()
    # Handle ```json ... ``` and ``` ... ```
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # No fences - strip any leading/trailing ``` just in case
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # 3. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Find the outermost JSON object or array
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(open_char)
        if start == -1:
            continue
        # Find matching close by counting nesting
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    json_str = cleaned[start:i+1]
                    # Try parsing as-is
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    # Fix common issues: trailing commas
                    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                    break

    # 5. Last resort: find first { and last } and try to fix truncation
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start == -1:
        start = cleaned.find('[')
        end = cleaned.rfind(']')
    if start != -1 and end > start:
        json_str = cleaned[start:end+1]
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try closing unclosed brackets
            opens = json_str.count('{') - json_str.count('}')
            if opens > 0:
                json_str += '}' * opens
            opens = json_str.count('[') - json_str.count(']')
            if opens > 0:
                json_str += ']' * opens
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return None


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gemma3:27b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Call LLM chat completions and return the assistant message content.
    Works with both Ollama and OpenAI backends.
    """
    resolved_model = _resolve_model(model)

    try:
        params: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if LLM_BACKEND == "openai":
            params.update(kwargs)

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return content if content is not None else ""

    except Exception as e:
        err = str(e)
        if LLM_BACKEND == "openai" and "insufficient_quota" in err:
            _QUOTA_FALLBACK = {"gpt-4o": "gpt-4o-mini", "gpt-4o-mini": "gpt-3.5-turbo"}
            if resolved_model in _QUOTA_FALLBACK:
                fallback = _QUOTA_FALLBACK[resolved_model]
                print(f"  Quota exceeded for {resolved_model}, retrying with {fallback}")
                return chat_completion(messages, fallback, temperature, max_tokens, **kwargs)
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
    Includes robust JSON extraction that handles markdown fences,
    preamble text, thinking tags, and minor formatting issues.
    """
    # Strengthen the JSON instruction in the system prompt
    modified_messages = []
    has_system = False
    for msg in messages:
        m = dict(msg)
        if m["role"] == "system":
            has_system = True
            m["content"] = m["content"].rstrip() + "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no text before or after the JSON."
        modified_messages.append(m)

    if not has_system:
        modified_messages.insert(0, {
            "role": "system",
            "content": "You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no text before or after the JSON."
        })

    # Also reinforce in the user message
    if modified_messages and modified_messages[-1]["role"] == "user":
        content = modified_messages[-1]["content"]
        if "json" not in content.lower()[-100:]:
            modified_messages[-1]["content"] = content + "\n\nRespond with ONLY valid JSON."

    resolved_model = _resolve_model(model)

    max_retries = 2
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        response_text = chat_completion(
            messages=modified_messages,
            model=resolved_model,
            temperature=temperature + (0.1 * (attempt - 1)),  # slightly raise temp on retry
            max_tokens=max_tokens,
            **kwargs,
        )
        last_response_text = response_text

        result = _extract_json(response_text)
        if result is not None:
            return result

        if attempt < max_retries:
            print(f"  WARNING: JSON parse failed (attempt {attempt}/{max_retries}), retrying...")

    # All attempts failed - log and return empty fallback
    print(f"  WARNING: Failed to parse JSON after {max_retries} attempts.")
    print(f"  Raw (first 300 chars): {last_response_text[:300]}")
    response_text = last_response_text
    if "linked_triples" in response_text or "triples" in response_text:
        return {"linked_triples": [], "triples": []}
    elif "entities" in response_text:
        return {"entities": []}
    elif "relations" in response_text or "relation" in response_text:
        return {"relations": [], "relations_found": []}
    elif "entity_groups" in response_text:
        return {"entity_groups": []}
    else:
        return {}


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Get an embedding vector for the given text."""
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        if LLM_BACKEND != "openai":
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            return [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), 512), 2)]
        raise Exception(f"Embedding API call failed: {str(e)}")
