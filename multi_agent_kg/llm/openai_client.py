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
import time
from dotenv import load_dotenv

load_dotenv()

# ── Backend configuration ─────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Timeout for Ollama calls (local models can be slow)
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # 5 min default
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "3.0"))
_RETRY_BACKOFF_CAP = float(os.getenv("LLM_RETRY_BACKOFF_CAP", "60.0"))
# When Ollama is unreachable (tunnel dropped, process restart), keep polling the
# /api/tags endpoint for this many seconds before giving up on the call.
_HEALTHCHECK_TIMEOUT = float(os.getenv("LLM_HEALTHCHECK_TIMEOUT", "300"))

if LLM_BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=_TIMEOUT,
    )

# Optional per-call usage logging. When LLM_USAGE_LOG points to a file path,
# each completion appends one JSON line with {model, prompt_tokens,
# completion_tokens, cached_tokens, reasoning_tokens}. Used for cost analysis.
_USAGE_LOG_PATH = os.getenv("LLM_USAGE_LOG")


def _log_usage(model: str, response: Any) -> None:
    if not _USAGE_LOG_PATH:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    record = {
        "ts": time.time(),
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cached_tokens": getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0,
    }
    try:
        with open(_USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _ollama_is_up() -> bool:
    """Quick TCP probe of the Ollama health endpoint. Returns False on any error."""
    if LLM_BACKEND == "openai":
        return True
    try:
        import urllib.request
        import urllib.error
        base = OLLAMA_BASE_URL.rstrip("/")
        # OLLAMA_BASE_URL commonly includes /v1; strip it for the native health endpoint.
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _wait_for_ollama(deadline: float) -> bool:
    """Poll Ollama until it responds or the deadline passes. Returns True on recovery."""
    while time.time() < deadline:
        if _ollama_is_up():
            return True
        time.sleep(5.0)
    return False

# Map old OpenAI model names → Ollama equivalents (for backward compat)
_OPENAI_TO_OLLAMA = {
    "gpt-4o": "gemma4:31b",
    "gpt-4o-mini": "gemma4:31b",
    "gpt-4": "gemma4:31b",
    "gpt-4-turbo": "gemma4:31b",
    "gpt-3.5-turbo": "gemma4:31b",
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


def _unwrap_json_mode_array(result: Any) -> Any:
    """Recover top-level arrays wrapped by JSON-object constrained decoding.

    Ollama/OpenAI JSON-object mode requires a top-level object. Some older
    prompts legitimately asked for a top-level array, so models often return
    ``{"items": [...]}``, ``{"domains": [...]}``, etc. Returning the sole
    list value preserves the old callsite contract without weakening JSON mode.
    """
    if isinstance(result, dict) and len(result) == 1:
        sole_value = next(iter(result.values()))
        if isinstance(sole_value, list):
            return sole_value
    return result


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gemma4:31b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Call LLM chat completions and return the assistant message content.
    Works with both Ollama and OpenAI backends.
    """
    resolved_model = _resolve_model(model)

    # GPT-5 / o-series reasoning models reject `max_tokens` and non-default `temperature`.
    is_reasoning = LLM_BACKEND == "openai" and (
        resolved_model.startswith("gpt-5")
        or resolved_model.startswith("o1")
        or resolved_model.startswith("o3")
        or resolved_model.startswith("o4")
    )

    params: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
    }
    if not is_reasoning:
        params["temperature"] = temperature
    if max_tokens is not None:
        if is_reasoning:
            # Reasoning models share max_completion_tokens between hidden reasoning
            # AND visible output. With low caps the model burns the whole budget on
            # reasoning and emits empty content. Inflate so visible output survives.
            params["max_completion_tokens"] = max(max_tokens * 4, 16384)
            # Keep reasoning lightweight so cost stays bounded for extraction tasks.
            params["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
        else:
            params["max_tokens"] = max_tokens
    # Ollama's OpenAI-compatible endpoint supports response_format for JSON
    # mode. Pass it through for both backends, but keep arbitrary provider
    # kwargs restricted to OpenAI so local calls don't receive unknown options.
    if "response_format" in kwargs:
        params["response_format"] = kwargs["response_format"]
    if LLM_BACKEND == "openai":
        params.update(kwargs)

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**params)
            _log_usage(resolved_model, response)
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            err = str(e)
            last_error = err
            if LLM_BACKEND == "openai" and "insufficient_quota" in err:
                _QUOTA_FALLBACK = {"gpt-4o": "gpt-4o-mini", "gpt-4o-mini": "gpt-3.5-turbo"}
                if resolved_model in _QUOTA_FALLBACK:
                    fallback = _QUOTA_FALLBACK[resolved_model]
                    print(f"  Quota exceeded for {resolved_model}, retrying with {fallback}")
                    return chat_completion(messages, fallback, temperature, max_tokens, **kwargs)

            transient_markers = [
                "Connection error",
                "ConnectError",
                "ReadError",
                "RemoteProtocolError",
                "timed out",
                "Timeout",
                "connection reset",
                "connection aborted",
                "connection refused",
                "temporarily unavailable",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "server disconnected",
                "EOF",
                "502",
                "503",
                "504",
            ]
            err_lower = err.lower()
            is_transient = any(marker.lower() in err_lower for marker in transient_markers)
            should_retry = attempt < _MAX_RETRIES and is_transient
            if should_retry:
                # Exponential backoff with jitter, capped — gives Ollama time to recover
                # from tunnel drops and model reloads instead of giving up in ~12s.
                base_sleep = min(_RETRY_BACKOFF * (2 ** (attempt - 1)), _RETRY_BACKOFF_CAP)
                # If the error looks like Ollama is down, actively wait for it to come back
                # up before the next attempt so we don't burn a retry on a still-dead server.
                connection_like = any(
                    m in err_lower for m in ("connection", "refused", "reset", "eof", "server disconnected")
                )
                if connection_like and LLM_BACKEND != "openai":
                    print(
                        f"  WARNING: Ollama appears unreachable on attempt {attempt}/{_MAX_RETRIES}; "
                        f"polling /api/tags for up to {_HEALTHCHECK_TIMEOUT:.0f}s before retrying"
                    )
                    deadline = time.time() + _HEALTHCHECK_TIMEOUT
                    recovered = _wait_for_ollama(deadline)
                    if recovered:
                        print(f"  Ollama came back up — retrying call")
                        continue
                    print(f"  Ollama still down after {_HEALTHCHECK_TIMEOUT:.0f}s; falling back to backoff sleep")
                print(
                    f"  WARNING: transient LLM failure on attempt {attempt}/{_MAX_RETRIES} "
                    f"for {resolved_model}; retrying in {base_sleep:.1f}s"
                )
                time.sleep(base_sleep)
                continue
            if "response_format" in params and any(
                marker in err_lower
                for marker in ("response_format", "unsupported", "unknown field", "invalid parameter")
            ):
                params.pop("response_format", None)
                print("  WARNING: JSON response_format unsupported by backend; retrying without constrained decoding")
                continue
            raise Exception(f"LLM API call failed ({resolved_model}): {err}")

    raise Exception(f"LLM API call failed ({resolved_model}): {last_error}")


def chat_completion_json(
    messages: List[Dict[str, str]],
    model: str = "gemma4:31b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    unwrap_array: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Call LLM requesting JSON output and parse the response.
    Includes robust JSON extraction that handles markdown fences,
    preamble text, thinking tags, and minor formatting issues.

    When ``unwrap_array=True``, a single-key dict-of-list response
    (e.g. ``{"items": [...]}``) is unwrapped to the inner list to
    support callsites that originally requested top-level arrays.
    Default is False so callers expecting dicts (e.g. ``{"sub_questions": [...]}``)
    are not silently stripped of their wrapper key.
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
            response_format={"type": "json_object"},
            **kwargs,
        )
        last_response_text = response_text

        result = _extract_json(response_text)
        if result is not None:
            return _unwrap_json_mode_array(result) if unwrap_array else result

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
