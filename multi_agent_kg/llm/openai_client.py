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


# ── Thinking-mode tag handling ────────────────────────────────────────
# Models like qwen3 and deepseek-r1 wrap reasoning in <think>...</think>
# and may wrap the actual answer in <answer>...</answer> tags.
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*)</answer>', re.DOTALL)


def _strip_thinking_tags(text: str) -> str:
    """Strip <think> blocks and unwrap <answer> tags from model responses.

    qwen3 models output: <think>reasoning</think><answer>actual output</answer>
    deepseek-r1 outputs: <think>reasoning</think>actual output
    """
    if not text:
        return text
    # 1) Remove <think>...</think> reasoning blocks
    text = _THINK_RE.sub('', text)
    # 2) Unwrap <answer>...</answer> — keep only the inner content
    m = _ANSWER_RE.search(text)
    if m:
        text = m.group(1)
    return text.strip()


def _is_qwen3_model(model: str) -> bool:
    """Check if model is a qwen3 variant (supports /no_think suffix in Ollama)."""
    return 'qwen3' in model.lower()


def _wrap_bare_list(result):
    """Wrap a bare JSON list in the expected dict structure.

    Many callers expect a dict like {"entities": [...]} or {"triples": [...]}.
    If the LLM returns a raw list, infer the wrapper key from the first element's
    fields and wrap it.  If the result is already a dict, return as-is.
    """
    if not isinstance(result, list):
        return result
    if not result:
        return result  # empty list — let caller handle
    sample = result[0]
    if not isinstance(sample, dict):
        return result
    # Infer the wrapper key from the shape of the objects
    if "text" in sample and ("type_guess" in sample or "type" in sample or "start" in sample):
        return {"entities": result}
    if "subject" in sample and "relation" in sample:
        return {"triples": result}
    if "relation_type" in sample and "definition" in sample:
        return {"relations_found": result}
    if "head_entity" in sample:
        return {"head_bindings": result}
    if "canonical_id" in sample or "canonical_name" in sample:
        return {"entity_groups": result}
    if "original" in sample and "normalized" in sample:
        return {"normalizations": result}
    if "text_span" in sample and "entity" in sample:
        return {"few_shot_examples": result}
    if "relation_type" in sample and "text_span" in sample:
        return {"relation_examples": result}
    if "linked_triples" in sample:
        return sample  # already a dict inside the list
    # Fallback: return as-is (caller must handle lists)
    return result


def _salvage_complete_json_objects(json_str: str):
    """Salvage complete JSON objects from a truncated array.

    Given a string like '[{"a":1},{"b":2},{"c":3...' (truncated),
    extract all complete objects and return them as a list.
    Returns None if no complete objects could be salvaged.
    """
    # Only works for arrays of objects
    if not json_str.strip().startswith("["):
        return None

    inner = json_str.strip()[1:]  # Remove leading [
    # Remove trailing ] if present
    if inner.rstrip().endswith("]"):
        inner = inner.rstrip()[:-1]

    complete_objects = []
    depth = 0
    obj_start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(inner):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_str = inner[obj_start:i+1]
                try:
                    obj = json.loads(obj_str)
                    complete_objects.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if complete_objects:
        return complete_objects
    return None


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
        content = content if content is not None else ""

        # Strip thinking tags from models that use them (qwen3, deepseek-r1)
        content = _strip_thinking_tags(content)

        return content

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
    resolved_model = _resolve_model(model)

    # ── Build messages with strong JSON enforcement ──────────────────
    modified_messages = []
    for msg in messages:
        modified_messages.append(msg.copy())

    # Enforce JSON output in the system message
    _JSON_SYSTEM_SUFFIX = (
        " You MUST respond with ONLY valid JSON. "
        "No markdown, no explanations, no commentary — just the raw JSON object or array."
    )
    if modified_messages and modified_messages[0]["role"] == "system":
        modified_messages[0]["content"] = modified_messages[0]["content"] + _JSON_SYSTEM_SUFFIX
    else:
        modified_messages.insert(0, {"role": "system", "content": _JSON_SYSTEM_SUFFIX.strip()})

    # Always append a JSON-only reminder at the end of the user prompt
    _JSON_USER_SUFFIX = (
        "\n\nIMPORTANT: Output ONLY the raw JSON object/array. "
        "Do NOT include any markdown formatting, code fences, headings, "
        "explanations, or text outside the JSON. Start your response with { or [."
    )
    if modified_messages and modified_messages[-1]["role"] == "user":
        modified_messages[-1]["content"] = modified_messages[-1]["content"] + _JSON_USER_SUFFIX

    # For qwen3 models on Ollama, disable thinking mode for JSON requests
    if LLM_BACKEND != "openai" and _is_qwen3_model(resolved_model):
        if modified_messages and modified_messages[-1]["role"] == "user":
            msg = modified_messages[-1]["content"]
            if "/no_think" not in msg:
                modified_messages[-1]["content"] = msg + " /no_think"

    # ── Call LLM with retry logic ────────────────────────────────────
    # Retry if the response is empty OR if the model returned prose instead of JSON
    response_text = ""
    for _attempt in range(3):
        response_text = chat_completion(
            messages=modified_messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        # Safety-net: strip any residual thinking/answer tags
        response_text = _strip_thinking_tags(response_text)
        stripped = response_text.strip()

        if not stripped:
            # Empty response — retry
            print(f"  [LLM] Empty response from {resolved_model}, retrying (attempt {_attempt + 2}/3)...")
            temperature = min(temperature + 0.1, 1.0)
            continue

        # Check if the response looks like JSON (starts with { or [ or ```)
        if stripped.startswith(("{", "[", "```")):
            break

        # Model returned prose/markdown instead of JSON — retry with stronger instruction
        if _attempt < 2:
            print(f"  [LLM] Non-JSON response from {resolved_model}, retrying (attempt {_attempt + 2}/3)...")
            # Replace the user suffix with an even stronger instruction for the retry
            if modified_messages and modified_messages[-1]["role"] == "user":
                content = modified_messages[-1]["content"]
                # Remove old suffix and add stronger one
                content = content.replace(_JSON_USER_SUFFIX, "")
                content = content + (
                    "\n\nCRITICAL: Your previous response was not valid JSON. "
                    "You MUST respond with ONLY a JSON object or array. "
                    "Start your response with { or [ and end with } or ]. "
                    "No text, no markdown, no explanations."
                )
                if "/no_think" not in content:
                    content = content + " /no_think"
                modified_messages[-1]["content"] = content
            temperature = max(temperature - 0.1, 0.0)
            continue
        break

    # ── Parse JSON ────────────────────────────────────────────────────
    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # --- Attempt 1: direct parse ---
    try:
        parsed = json.loads(cleaned)
        return _wrap_bare_list(parsed)
    except json.JSONDecodeError:
        pass

    # --- Attempt 2: extract JSON structure from surrounding text ---
    try:
        arr_start = cleaned.find("[")
        obj_start = cleaned.find("{")

        # Prefer the outermost structure (whichever comes first)
        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            start = arr_start
            end = cleaned.rfind("]") + 1
            if end <= start:
                # Truncated array — no closing ]
                end = cleaned.rfind("}") + 1
                if end > start:
                    json_str = cleaned[start:end] + "]"
                    json_str = re.sub(r',\s*\]$', ']', json_str)
                else:
                    json_str = None
            else:
                json_str = cleaned[start:end]
        elif obj_start != -1:
            start = obj_start
            end = cleaned.rfind("}") + 1
            json_str = cleaned[start:end] if end > start else None
        else:
            json_str = None

        if json_str:
            # Try fix strategies
            for attempt in [
                json_str,
                json_str.replace(",]", "]").replace(",}", "}"),
                re.sub(r',(\s*[}\]])', r'\1', json_str),
            ]:
                try:
                    parsed = json.loads(attempt)
                    return _wrap_bare_list(parsed)
                except json.JSONDecodeError:
                    continue

            # Salvage complete objects from truncated array or dict-wrapped array
            # Works for both bare arrays "[{...},..." and dict-wrapped arrays '{"entities":[{...},...'
            salvaged = _salvage_complete_json_objects(json_str)
            if salvaged is not None:
                return _wrap_bare_list(salvaged)
            # Also try extracting the inner array from a dict wrapper
            inner_bracket = json_str.find("[")
            if inner_bracket > 0:  # > 0 means there IS an inner [ that's not the start
                inner_array_str = json_str[inner_bracket:]
                salvaged = _salvage_complete_json_objects(inner_array_str)
                if salvaged is not None:
                    return _wrap_bare_list(salvaged)

            # Try closing unclosed brackets — close INNER brackets first
            # Arrays ([) are typically nested inside objects ({), so close ] before }
            repaired = json_str
            if repaired.count("[") > repaired.count("]"):
                repaired += "]" * (repaired.count("[") - repaired.count("]"))
            if repaired.count("{") > repaired.count("}"):
                repaired += "}" * (repaired.count("{") - repaired.count("}"))
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            try:
                parsed = json.loads(repaired)
                return _wrap_bare_list(parsed)
            except Exception:
                pass
    except Exception:
        pass

    # --- Last resort: return empty structure based on prompt content ---
    print(f"  WARNING: Failed to parse JSON, returning empty result.")
    print(f"  Raw response (first 500 chars): {response_text[:500]}")
    # Inspect the original prompt to guess the expected structure
    prompt_text = ""
    for msg in messages:
        prompt_text += msg.get("content", "")
    if "linked_triples" in prompt_text:
        return {"linked_triples": [], "triples": []}
    elif "head_bindings" in prompt_text:
        return {"head_bindings": []}
    elif "triples" in prompt_text:
        return {"triples": []}
    elif "entity_groups" in prompt_text:
        return {"entity_groups": []}
    elif "entities" in prompt_text:
        return {"entities": []}
    elif "relations_found" in prompt_text:
        return {"relations_found": []}
    elif "relations" in prompt_text:
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
