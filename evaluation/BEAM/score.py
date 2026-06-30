"""
BEAM benchmark evaluator — Phase 2: Scoring.

Loads the responses file produced by run_eval.py and scores each answer
using BEAM's LLM-as-judge protocol (unified_llm_judge_base_prompt).

Judge model is configurable — defaults to LLM_DEFAULT_MODEL but you
should point it at a strong reasoning model for comparable results:
  - Qwen3.7 Plus  (recommended, good cost-quality)
  - DeepSeek-V4-Pro (max quality)
  - gpt-4o-mini   (matches published BEAM paper baselines)

Usage:
    # Score responses with default model
    python evaluation/BEAM/score.py \
        --responses evaluation/results/beam_100K_responses.json \
        --output evaluation/results/beam_100K_scores.json

    # Use specific judge model (override)
    python evaluation/BEAM/score.py \
        --responses evaluation/results/beam_100K_responses.json \
        --judge-model Qwen/Qwen3.7-Plus \
        --output evaluation/results/beam_100K_scores.json

    # Resume interrupted scoring (skips already-scored questions)
    python evaluation/BEAM/score.py \
        --responses evaluation/results/beam_100K_responses.json \
        --output evaluation/results/beam_100K_scores.json \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("beam_score")

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Judge LLM wrapper (LangChain-style .invoke(prompt).content interface)
# backed by the project's OpenAI-compatible client)
# ---------------------------------------------------------------------------

class _FakeContent:
    def __init__(self, text: str):
        self.content = text


class JudgeLLM:
    """
    Thin wrapper that mimics langchain's `.invoke(prompt).content` interface
    using the project's existing openai_client.
    """

    def __init__(self, model: Optional[str] = None):
        from multi_agent_kg.llm.openai_client import client, DEFAULT_CHAT_MODEL
        self._model = model or os.environ.get("BEAM_JUDGE_MODEL", DEFAULT_CHAT_MODEL)
        logger.info("Judge model: %s", self._model)

        # If the judge model is on Fireworks, use a Fireworks client instead of
        # the default vLLM/ollama client.
        if self._model.startswith("accounts/fireworks/"):
            fw_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("FIREWORKS_API_KEY", "")
            if not fw_key:
                # Try reading from ~/.fireworks_api_key
                import pathlib
                key_file = pathlib.Path.home() / ".fireworks_api_key"
                if key_file.exists():
                    fw_key = key_file.read_text().strip()
            if not fw_key:
                raise RuntimeError(
                    "Judge model is on Fireworks but no API key found. "
                    "Set EMBEDDING_API_KEY or FIREWORKS_API_KEY in .env, "
                    "or save key to ~/.fireworks_api_key"
                )
            from openai import OpenAI as _OpenAI
            self._client = _OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=fw_key,
                timeout=120,
            )
            logger.info("Judge client: Fireworks (api.fireworks.ai)")
        else:
            self._client = client
            logger.info("Judge client: %s", os.getenv("LLM_BACKEND", "ollama"))

    def invoke(self, prompt: str, max_retries: int = 5) -> _FakeContent:
        """Call the judge model with exponential backoff."""
        delay = 2.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    # Generous budget: a thinking judge (e.g. deepseek-v4) spends
                    # tokens on CoT before the {"score",...} JSON. 2048 could
                    # truncate mid-reasoning and silently score 0.0; bumped to
                    # 4096 so long CoT chains still leave room for the JSON
                    # verdict ({"score":..., "reasoning":...} is ~200 tokens).
                    max_tokens=4096,
                )
                return _FakeContent(resp.choices[0].message.content or "")
            except Exception as exc:
                last_exc = exc
                logger.warning("Judge call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"Judge exhausted retries: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# BEAM judge prompt (from src/prompts.py in the BEAM repo)
# ---------------------------------------------------------------------------

UNIFIED_LLM_JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).
2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element
4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.
5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


# ---------------------------------------------------------------------------
# JSON parsing (robust — handles CoT models that wrap JSON in markdown)
# ---------------------------------------------------------------------------

def _parse_judge_response(text: str) -> Dict:
    """Extract JSON from judge response, stripping markdown fences / CoT preamble."""
    text = text.strip()

    # Strip markdown code block
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find last JSON object in the text (CoT models may emit reasoning first)
    matches = list(re.finditer(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass

    # Try json_repair if available
    try:
        from json_repair import repair_json
        return json.loads(repair_json(text))
    except Exception:
        pass

    logger.warning("Could not parse judge response: %r", text[:200])
    return {"score": 0.0, "reason": f"parse_error: {text[:100]}"}


# ---------------------------------------------------------------------------
# Per question-type scoring (mirrors BEAM compute_metrics.py evaluate_* fns)
# All types use unified_llm_judge_base_prompt + optional extra metrics.
# ---------------------------------------------------------------------------

def _score_rubric_items(
    rubric: List[str],
    llm_response: str,
    question: str,
    judge: JudgeLLM,
) -> Dict[str, Any]:
    """Score a list of rubric items and return aggregated result."""
    judge_responses = []
    total = 0.0

    for item in rubric:
        prompt = (UNIFIED_LLM_JUDGE_PROMPT
                  .replace("<rubric_item>", str(item))
                  .replace("<llm_response>", llm_response)
                  .replace("<question>", question))
        raw = judge.invoke(prompt).content.strip()
        parsed = _parse_judge_response(raw)
        score = float(parsed.get("score", 0.0))
        total += score
        judge_responses.append(parsed)

    llm_judge_score = total / len(rubric) if rubric else 0.0
    return {"llm_judge_score": llm_judge_score, "llm_judge_responses": judge_responses}


def score_question(
    q_type: str,
    question: str,
    rubric: Any,
    llm_response: str,
    judge: JudgeLLM,
) -> Dict[str, Any]:
    """
    Score one probing question of a given type.

    rubric can be:
      - list of strings  (most types)
      - list of lists    (event_ordering, summarization use nested structure)
    """
    if not rubric:
        return {"llm_judge_score": 0.0, "llm_judge_responses": [], "_skipped": "empty_rubric"}
    if not llm_response:
        return {"llm_judge_score": 0.0, "llm_judge_responses": [], "_skipped": "empty_response"}

    # Flatten nested rubric to list of strings
    flat_rubric: List[str] = []
    for item in rubric:
        if isinstance(item, list):
            flat_rubric.extend(str(x) for x in item)
        else:
            flat_rubric.append(str(item))

    if not flat_rubric:
        return {"llm_judge_score": 0.0, "llm_judge_responses": [], "_skipped": "empty_rubric"}

    result = _score_rubric_items(flat_rubric, llm_response, question, judge)

    # event_ordering also computes Kendall-tau via semantic alignment
    if q_type == "event_ordering":
        try:
            result["_kendall_tau"] = _event_ordering_tau(rubric, llm_response)
        except Exception as exc:
            logger.debug("Kendall-tau failed: %s", exc)

    return result


def _event_ordering_tau(rubric_list, llm_response: str) -> float:
    """Approximate Kendall-tau using sentence-transformers cosine alignment."""
    try:
        from sentence_transformers import SentenceTransformer, util
        from scipy.stats import kendalltau
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        ref = [str(x) for x in rubric_list]
        sys_list = [s.strip() for s in llm_response.split("\n") if s.strip()]
        if not ref or not sys_list:
            return 0.0

        ref_emb = model.encode(ref, normalize_embeddings=True)
        sys_emb = model.encode(sys_list, normalize_embeddings=True)

        used = set()
        sys_canon = []
        for sv in sys_emb:
            sims = util.cos_sim(sv, ref_emb)[0].numpy()
            best = int(np.argmax(sims))
            if sims[best] >= 0.65 and best not in used:
                sys_canon.append(ref[best])
                used.add(best)
            else:
                sys_canon.append(sys_list[len(sys_canon)])

        union = list(dict.fromkeys(ref + sys_canon))
        tie_rank = len(union) + 1

        def to_rank(seq):
            r = {item: i + 1 for i, item in enumerate(seq)}
            return [r.get(u, tie_rank) for u in union]

        tau, _ = kendalltau(to_rank(ref), to_rank(sys_canon), variant="b")
        return float((tau + 1) / 2) if tau is not None else 0.0
    except ImportError:
        return 0.0


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def aggregate_scores(chats: List[Dict]) -> Dict[str, Any]:
    """Aggregate llm_judge_score across all chats and question types."""
    by_type: Dict[str, List[float]] = {}
    all_scores: List[float] = []

    for chat in chats:
        scored = chat.get("scored_responses", {})
        for q_type, questions in scored.items():
            for q in questions:
                score = q.get("_score", {}).get("llm_judge_score", None)
                if score is None:
                    continue
                by_type.setdefault(q_type, []).append(score)
                all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    per_type = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())}

    return {
        "overall": round(overall, 4),
        "n_questions": len(all_scores),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def run_scoring(
    responses_path: Path,
    judge: JudgeLLM,
    output_path: Path,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Load responses, score each answer with the judge, write scored output.

    With --resume, loads existing output and skips already-scored entries.
    """
    data = json.loads(responses_path.read_text())
    meta = data.get("meta", {})
    chats = data.get("chats", [])

    # Load existing scored output for resume
    existing_scores: Dict[str, Dict[str, List]] = {}
    if resume and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            for c in existing.get("chats", []):
                existing_scores[str(c["conversation_id"])] = c.get("scored_responses", {})
            logger.info("Resume: loaded %d already-scored chats", len(existing_scores))
        except Exception as exc:
            logger.warning("Could not load resume file: %s", exc)

    scored_chats = []

    for chat_idx, chat in enumerate(chats):
        conv_id = str(chat.get("conversation_id", f"chat_{chat_idx}"))
        logger.info("=== Scoring chat %d/%d — ID: %s ===", chat_idx + 1, len(chats), conv_id)

        responses = chat.get("responses", {})
        scored_responses: Dict[str, List] = {}

        for q_type, questions in responses.items():
            type_scored = []
            existing_type = existing_scores.get(conv_id, {}).get(q_type, [])

            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                rubric = q_item.get("rubric", [])
                llm_response = q_item.get("llm_response", "")

                # Resume: reuse existing score if present
                if resume and q_idx < len(existing_type):
                    existing_q = existing_type[q_idx]
                    if "_score" in existing_q:
                        type_scored.append(existing_q)
                        logger.debug("  [%s] Q%d — reused cached score", q_type, q_idx + 1)
                        continue

                logger.info("  [%s] Q%d: %r", q_type, q_idx + 1, question[:70])

                q_score = score_question(
                    q_type=q_type,
                    question=question,
                    rubric=rubric,
                    llm_response=llm_response,
                    judge=judge,
                )

                entry = dict(q_item)
                entry["_score"] = q_score
                type_scored.append(entry)

                judge_score = q_score.get("llm_judge_score", 0.0)
                logger.info("  → judge_score=%.3f", judge_score)

            scored_responses[q_type] = type_scored

        scored_chat = dict(chat)
        scored_chat["scored_responses"] = scored_responses
        scored_chats.append(scored_chat)

        # Save incrementally
        _save_scored(output_path, scored_chats, meta, judge._model)

    agg = aggregate_scores(scored_chats)
    result = {
        "meta": {**meta, "judge_model": judge._model},
        "aggregate": agg,
        "chats": scored_chats,
    }
    _save_scored(output_path, scored_chats, meta, judge._model)
    return result


def _save_scored(path: Path, chats: List[Dict], meta: Dict, judge_model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate_scores(chats)
    out = {
        "meta": {**meta, "judge_model": judge_model},
        "aggregate": agg,
        "chats": chats,
    }
    path.write_text(json.dumps(out, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEAM Phase 2: Score responses with LLM judge"
    )
    parser.add_argument(
        "--responses", required=True,
        help="Path to responses JSON produced by run_eval.py"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for scored results (default: same dir as responses, _scores suffix)"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge model name (overrides BEAM_JUDGE_MODEL env and LLM_DEFAULT_MODEL). "
             "Recommended: Qwen/Qwen3.7-Plus or deepseek-ai/DeepSeek-V4-Pro or gpt-4o-mini"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted scoring (skip already-scored questions)"
    )
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Responses file not found: {responses_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = responses_path.parent / responses_path.name.replace("_responses", "_scores")
        if output_path == responses_path:
            output_path = responses_path.parent / (responses_path.stem + "_scores.json")

    if args.judge_model:
        os.environ["BEAM_JUDGE_MODEL"] = args.judge_model

    judge = JudgeLLM(model=args.judge_model)

    logger.info("BEAM Evaluation — Phase 2: Scoring")
    logger.info("Responses:   %s", responses_path)
    logger.info("Judge model: %s", judge._model)
    logger.info("Output:      %s", output_path)
    logger.info("Resume:      %s", args.resume)

    result = run_scoring(
        responses_path=responses_path,
        judge=judge,
        output_path=output_path,
        resume=args.resume,
    )

    print(f"\n{'='*60}")
    print("BEAM Phase 2 Complete — Aggregate Scores")
    print(f"{'='*60}")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"\nScored output saved to: {output_path}")


if __name__ == "__main__":
    main()
