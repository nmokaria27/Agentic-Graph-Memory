"""Utilities for small HotpotQA pilot experiments."""

from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class HotpotExample:
    question_id: str
    question: str
    answer: str
    contexts: List[Dict[str, Any]]
    supporting_facts: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, and normalize whitespace."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def answer_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def context_text(context: Dict[str, Any]) -> str:
    sentences = context.get("sentences", [])
    if sentences and isinstance(sentences[0], list):
        return " ".join(" ".join(sentence) for sentence in sentences)
    return " ".join(str(sentence) for sentence in sentences)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "document"


def load_hotpot_json(path: Path) -> List[HotpotExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for item in raw:
        contexts = [
            {"title": title, "sentences": sentences}
            for title, sentences in item.get("context", [])
        ]
        examples.append(
            HotpotExample(
                question_id=item.get("_id") or item.get("id"),
                question=item["question"],
                answer=item.get("answer", ""),
                contexts=contexts,
                supporting_facts=item.get("supporting_facts", []),
            )
        )
    return examples


def select_pilot_examples(examples: Iterable[HotpotExample], n: int) -> List[HotpotExample]:
    selected = []
    yes_no_count = 0
    for example in examples:
        answer = (example.answer or "").strip()
        if not answer:
            continue
        if len(answer.split()) > 8:
            continue
        is_yes_no = normalize_answer(answer) in {"yes", "no"}
        if is_yes_no and yes_no_count >= max(2, n // 5):
            continue
        if is_yes_no:
            yes_no_count += 1
        selected.append(example)
        if len(selected) >= n:
            break
    return selected


def save_examples(examples: List[HotpotExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([example.to_dict() for example in examples], indent=2), encoding="utf-8")


def load_prepared_examples(path: Path) -> List[HotpotExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [HotpotExample(**item) for item in raw]


def write_context_corpus(examples: List[HotpotExample], output_dir: Path) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_titles = set()
    manifest = []
    for example in examples:
        for context in example.contexts:
            title = context.get("title", "document")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = context_text(context)
            filename = f"{_slug(title)}.txt"
            (output_dir / filename).write_text(text, encoding="utf-8")
            manifest.append({"title": title, "filename": filename, "text": text})
    return manifest


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3
    }


def lexical_retrieve(question: str, contexts: List[Dict[str, Any]], k: int = 4) -> List[Dict[str, Any]]:
    query_terms = _terms(question)
    ranked = sorted(
        contexts,
        key=lambda context: (
            len(query_terms & _terms(context.get("title", ""))),
            len(query_terms & _terms(context_text(context))),
        ),
        reverse=True,
    )
    return ranked[:k]


def answer_metrics(prediction: str, gold: str) -> Dict[str, float]:
    prediction = clean_prediction_for_scoring(prediction, gold)
    return {
        "exact_match": answer_exact_match(prediction, gold),
        "token_f1": answer_token_f1(prediction, gold),
    }


def clean_prediction_for_scoring(prediction: str, gold: str) -> str:
    """Normalize generated prose to a short-answer prediction for HotpotQA scoring."""
    text = (prediction or "").strip()
    if not text:
        return ""
    text = re.sub(r"\[Data:[^\]]+\]", "", text).strip()
    text = re.sub(r"\[[0-9,\s]+\]", "", text).strip()
    gold_norm = normalize_answer(gold)
    lowered = text.lower().strip()
    if gold_norm in {"yes", "no"}:
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    for prefix in ("the answer is", "answer:", "short answer:", "final answer:"):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            lowered = text.lower()
            break
    return text.strip().strip('"').strip("'").rstrip(".")


def aggregate_answer_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"num_questions": 0, "exact_match": 0.0, "token_f1": 0.0, "answer_rate": 0.0}
    return {
        "num_questions": len(rows),
        "exact_match": round(sum(row["metrics"]["exact_match"] for row in rows) / len(rows), 4),
        "token_f1": round(sum(row["metrics"]["token_f1"] for row in rows) / len(rows), 4),
        "answer_rate": round(sum(1 for row in rows if row.get("answer", "").strip()) / len(rows), 4),
    }
