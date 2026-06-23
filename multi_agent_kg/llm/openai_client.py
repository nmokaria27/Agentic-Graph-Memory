"""
LLM client wrapper supporting Ollama (default), OpenAI, and VLLM backends.

Backend selection:
- Set LLM_BACKEND=openai  to use OpenAI (requires OPENAI_API_KEY)
- Set LLM_BACKEND=vllm    to use a VLLM server (set VLLM_BASE_URL, e.g. http://gpu-host:8000/v1)
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
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# Default chat model for callsites that don't pass one explicitly. Follows
# LLM_DEFAULT_MODEL so vLLM/OpenAI deployments aren't hit with the Ollama name.
DEFAULT_CHAT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")

# Timeout for local/remote model calls
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # 5 min default
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "3.0"))
_RETRY_BACKOFF_CAP = float(os.getenv("LLM_RETRY_BACKOFF_CAP", "60.0"))
# When Ollama/VLLM is unreachable (tunnel dropped, process restart), keep polling
# the health endpoint for this many seconds before giving up on the call.
_HEALTHCHECK_TIMEOUT = float(os.getenv("LLM_HEALTHCHECK_TIMEOUT", "300"))

if LLM_BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_BACKEND == "vllm":
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout=_TIMEOUT,
    )
else:
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=_TIMEOUT,
    )

# ── Embedding backend (may differ from chat) ──────────────────────────
# Embeddings can target a separate OpenAI-compatible server than chat. Common
# setup: chat on vLLM (gemma), embeddings on Ollama's mxbai-embed-large via the
# existing tunnel. Set EMBEDDING_BASE_URL to route embedding calls elsewhere;
# otherwise the main chat client is reused.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
if EMBEDDING_BASE_URL:
    embed_client = OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=os.getenv("EMBEDDING_API_KEY", "ollama"),
        timeout=_TIMEOUT,
    )
else:
    embed_client = client

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


def _backend_is_up() -> bool:
    """Quick TCP probe of the active backend health endpoint. Returns False on any error."""
    if LLM_BACKEND == "openai":
        return True
    try:
        import urllib.request
        if LLM_BACKEND == "vllm":
            # VLLM exposes /health on its HTTP server
            base = VLLM_BASE_URL.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            url = f"{base}/health"
        else:
            base = OLLAMA_BASE_URL.rstrip("/")
            # OLLAMA_BASE_URL commonly includes /v1; strip it for the native health endpoint.
            if base.endswith("/v1"):
                base = base[:-3]
            url = f"{base}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


# Keep old name as alias so any external callers are not broken
_ollama_is_up = _backend_is_up


def _wait_for_ollama(deadline: float) -> bool:
    """Poll the active backend until it responds or the deadline passes. Returns True on recovery."""
    while time.time() < deadline:
        if _backend_is_up():
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
    """Resolve model name: translate OpenAI names to Ollama/VLLM when using those backends."""
    if LLM_BACKEND in ("openai", "vllm"):
        return model
    return _OPENAI_TO_OLLAMA.get(model, model)


# Model families known to emit <think>...</think> reasoning tokens.
# On Ollama: GBNF json_object grammar blocks the leading '<' → empty content.
# On VLLM:  json_object mode strips special tokens; thinking models may still
#            embed reasoning tokens in the visible output, so we skip constrained
#            decoding for them and rely on our robust _extract_json parser.
_THINKING_MODEL_PATTERNS = (
    "gemma4",       # Google Gemma 4 family (gemma4:12b, gemma4:27b, gemma4:31b)
    "deepseek-r1",  # DeepSeek R1 family
    "qwen3",        # Qwen 3 (thinking by default unless /no_think)
    "qwq",          # Qwen QwQ reasoning models
    "llama-3.3",    # Llama 3.3 instruct (sometimes emits chain-of-thought preamble)
)


def _model_is_thinking(model_name: str) -> bool:
    """Return True if the model is known to emit reasoning/thinking tokens."""
    name_lower = model_name.lower()
    return any(pat in name_lower for pat in _THINKING_MODEL_PATTERNS)


def _extract_json(text: str) -> Any:
    """
    Robustly extract and parse JSON from LLM output that may contain
    markdown fences, preamble text, thinking tags, or trailing commentary.

    Handles thinking-model output where:
    - <think>...</think> blocks precede the JSON (possibly unclosed if truncated)
    - JSON may be truncated mid-value due to token limits
    - Markdown fences may or may not be present
    """
    if not text or not text.strip():
        return None

    # 1. Strip <think>...</think> blocks (closed tags)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Strip unclosed <think> tags (model hit token limit mid-reasoning, then
    #    continued with JSON, OR the JSON follows a truncated think block)
    if '<think>' in cleaned:
        # Take everything after the last <think> tag that was never closed
        parts = cleaned.split('<think>')
        # The JSON is most likely in the last part, after any remaining think content
        # Look for JSON start in each part from the end
        for part in reversed(parts):
            if '{' in part or '[' in part:
                cleaned = part.strip()
                break
        else:
            cleaned = parts[-1].strip()

    # 3. Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Handle unclosed fences (truncation cut off the closing ```)
        open_fence = re.search(r'```(?:json)?\s*\n?', cleaned)
        if open_fence:
            cleaned = cleaned[open_fence.end():].strip()
        # Strip any leading/trailing ``` fragments
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # 4. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 5. Find the outermost JSON object or array via bracket matching
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(open_char)
        if start == -1:
            continue
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
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                    break

    # 6. Truncation repair: model hit token limit mid-JSON.
    #    Find the first { and attempt to close the structure.
    result = _repair_truncated_json(cleaned)
    if result is not None:
        return result

    return None


def _repair_truncated_json(text: str) -> Any:
    """Attempt to salvage a truncated JSON object/array.

    Strategy: find the outermost opening brace, take everything from there to the
    end, strip any trailing partial value (e.g. ``"type": "PEO``), then close all
    open brackets/braces. This recovers all *complete* entries in a truncated list.
    """
    start = text.find('{')
    if start == -1:
        start = text.find('[')
    if start == -1:
        return None

    fragment = text[start:]

    # Strip trailing commas
    fragment = re.sub(r',(\s*[}\]])', r'\1', fragment)

    # Try parsing as-is first (maybe it's valid after comma cleanup)
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # Truncation likely cut mid-value. Trim back to the last complete JSON element.
    # Strategy: remove the last partial key-value or array element.
    # Find the last complete "}" or "]" or quoted string followed by comma/bracket.
    # Simpler approach: walk backwards from the end, remove characters until we can
    # close the brackets and parse.

    # First, try removing everything after the last comma at depth > 0
    # (this drops the truncated element but keeps all complete ones)
    trimmed = _trim_to_last_complete_element(fragment)
    if trimmed:
        # Count unclosed brackets/braces and close them
        opens_brace = trimmed.count('{') - trimmed.count('}')
        opens_bracket = trimmed.count('[') - trimmed.count(']')
        suffix = ']' * max(opens_bracket, 0) + '}' * max(opens_brace, 0)
        candidate = trimmed + suffix
        # Fix trailing commas that may appear before the new closing brackets
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Brute force: close all open brackets/braces on the raw fragment
    opens_brace = fragment.count('{') - fragment.count('}')
    opens_bracket = fragment.count('[') - fragment.count(']')
    if opens_brace > 0 or opens_bracket > 0:
        candidate = fragment + ']' * max(opens_bracket, 0) + '}' * max(opens_brace, 0)
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _trim_to_last_complete_element(fragment: str) -> Optional[str]:
    """Walk backwards through a JSON fragment to find the end of the last
    complete element (object or value). Returns the trimmed string, or None."""
    # Find positions of all }, ], and complete quoted strings followed by
    # structural characters. We look for the last "}" that reduces depth,
    # or the last complete array element.

    # Simple heuristic: find the last occurrence of "},", "}", "],", "]",
    # or a complete value followed by "," and trim there.
    # Look for the last "}" or "]" that appears before a "," or at the end
    best = -1
    for pattern in [r'\}\s*,', r'\]\s*,', r'"\s*,', r'\d\s*,', r'true\s*,', r'false\s*,', r'null\s*,']:
        for m in re.finditer(pattern, fragment):
            end_pos = m.end() - 1  # position of the comma
            if end_pos > best:
                best = end_pos
    if best > 0:
        return fragment[:best]

    # No comma found — try last complete closing bracket
    for i in range(len(fragment) - 1, -1, -1):
        if fragment[i] in ('}', ']'):
            return fragment[:i+1]

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
    model: str = DEFAULT_CHAT_MODEL,
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
    # VLLM-hosted models always accept the standard parameters.
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
        elif _model_is_thinking(resolved_model) and LLM_BACKEND not in ("openai",):
            # Thinking models (gemma4, deepseek-r1, qwen3, etc.) burn completion
            # tokens on hidden <think> blocks. Inflate max_tokens so the visible
            # JSON output isn't truncated after reasoning consumes the budget.
            # This applies to both Ollama and VLLM backends.
            params["max_tokens"] = max(max_tokens * 3, 8192)
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
                    _backend_label = "VLLM" if LLM_BACKEND == "vllm" else "Ollama"
                    print(
                        f"  WARNING: {_backend_label} appears unreachable on attempt {attempt}/{_MAX_RETRIES}; "
                        f"polling health endpoint for up to {_HEALTHCHECK_TIMEOUT:.0f}s before retrying"
                    )
                    deadline = time.time() + _HEALTHCHECK_TIMEOUT
                    recovered = _wait_for_ollama(deadline)
                    if recovered:
                        print(f"  {_backend_label} came back up — retrying call")
                        continue
                    print(f"  {_backend_label} still down after {_HEALTHCHECK_TIMEOUT:.0f}s; falling back to backoff sleep")
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
    model: str = DEFAULT_CHAT_MODEL,
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

    # Detect thinking models that emit <think>...</think> before the answer.
    # On Ollama: GBNF json_object grammar blocks <think> tokens → empty content.
    # On VLLM:  json_object mode is generally safe, but thinking models may still
    #           embed reasoning in visible output; skip constrained decoding to be safe.
    _is_thinking_model = _model_is_thinking(resolved_model)

    max_retries = 3
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        # Use json_object mode only when it won't conflict with model behavior:
        # - Always for OpenAI (native support)
        # - Never for thinking models on Ollama or VLLM (may conflict with reasoning tokens)
        # - Only on first attempt for non-thinking Ollama/VLLM models (fallback on retry)
        if LLM_BACKEND == "openai":
            use_json_mode = True
        elif _is_thinking_model:
            use_json_mode = False
        else:
            use_json_mode = (attempt == 1)

        call_kwargs = dict(kwargs)
        if use_json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}

        response_text = chat_completion(
            messages=modified_messages,
            model=resolved_model,
            temperature=temperature + (0.1 * (attempt - 1)),
            max_tokens=max_tokens,
            **call_kwargs,
        )
        last_response_text = response_text

        # Empty response with json_object mode → GBNF conflict, skip to unconstrained retry
        if not response_text.strip() and use_json_mode:
            print(f"  WARNING: Empty response with json_object mode (attempt {attempt}/{max_retries}); retrying without constrained decoding...")
            continue

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


# ── Embeddings ────────────────────────────────────────────────────────
# Backend-aware default embedding model. mxbai-embed-large is asymmetric:
# retrieval queries must be prefixed, passages are embedded raw.
# VLLM: set EMBEDDING_MODEL to the embedding model served by your VLLM instance
# (e.g. "intfloat/e5-mistral-7b-instruct" or "BAAI/bge-large-en-v1.5").
# Embeddings run on OpenAI only when LLM_BACKEND=openai AND no separate
# EMBEDDING_BASE_URL is set; otherwise they hit Ollama/vLLM (mxbai default).
_EMBED_ON_OPENAI = LLM_BACKEND == "openai" and not EMBEDDING_BASE_URL
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small" if _EMBED_ON_OPENAI
    else "mxbai-embed-large",
)

_MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# mxbai-embed-large has a 512-token context window; truncate conservatively.
# For VLLM-served models with larger windows, increase EMBEDDING_MAX_CHARS.
_EMBED_MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "1800"))
# GPU batching (embeddings served by vLLM) is much faster than Ollama CPU; bump
# default batch size only when embeddings actually run on vLLM. When embeddings
# are routed to a separate server (EMBEDDING_BASE_URL, e.g. Ollama), stay small.
_EMBED_ON_VLLM = LLM_BACKEND == "vllm" and not EMBEDDING_BASE_URL
_EMBED_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256" if _EMBED_ON_VLLM else "64"))


def _truncate_for_embedding(text: str) -> str:
    text = text.strip()
    if len(text) > _EMBED_MAX_CHARS:
        return text[:_EMBED_MAX_CHARS]
    return text


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Get an embedding vector for the given text."""
    return get_embeddings([text], model=model)[0]


def get_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Get embedding vectors for a batch of texts (order preserved)."""
    model = model or DEFAULT_EMBEDDING_MODEL
    cleaned = [_truncate_for_embedding(t) or " " for t in texts]
    vectors: List[List[float]] = []
    for start in range(0, len(cleaned), _EMBED_BATCH_SIZE):
        batch = cleaned[start:start + _EMBED_BATCH_SIZE]
        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = embed_client.embeddings.create(model=model, input=batch)
                # API may return items out of order; sort by index.
                items = sorted(response.data, key=lambda item: item.index)
                vectors.extend([item.embedding for item in items])
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    delay = min(_RETRY_BACKOFF * attempt, _RETRY_BACKOFF_CAP)
                    print(
                        f"  WARNING: Embedding call failed (attempt {attempt}/{_MAX_RETRIES}): "
                        f"{exc}; retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
        if last_error is not None:
            raise Exception(f"Embedding API call failed: {last_error}")
    return vectors


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """Embed a retrieval query (applies the mxbai query prefix when needed)."""
    model = model or DEFAULT_EMBEDDING_MODEL
    if "mxbai" in model:
        text = _MXBAI_QUERY_PREFIX + text
    return get_embedding(text, model=model)


def embed_passages(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Embed passages/documents for indexing (no query prefix)."""
    return get_embeddings(texts, model=model)
