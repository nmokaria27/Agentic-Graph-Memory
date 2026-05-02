"""Utilities for MuSiQue-style multi-hop QA pilot experiments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class MusiqueExample:
    question_id: str
    question: str
    answer: str
    contexts: List[Dict[str, Any]]
    supporting_facts: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug or "document"


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if stripped.startswith("["):
        return json.loads(raw_text)
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _paragraph_text(paragraph: Dict[str, Any]) -> str:
    for key in ("paragraph_text", "text", "content", "passage"):
        value = paragraph.get(key)
        if isinstance(value, str):
            return value
    sentences = paragraph.get("sentences")
    if isinstance(sentences, list):
        return " ".join(str(sentence) for sentence in sentences)
    return ""


def _paragraph_title(paragraph: Dict[str, Any], index: int) -> str:
    for key in ("title", "paragraph_title", "wiki_title", "id"):
        value = paragraph.get(key)
        if value:
            return str(value)
    return f"paragraph_{index}"


def load_musique_json(path: Path) -> List[MusiqueExample]:
    raw = _read_json_or_jsonl(path)
    examples: List[MusiqueExample] = []
    for item_index, item in enumerate(raw, start=1):
        answer = item.get("answer")
        if isinstance(answer, dict):
            answer = answer.get("text") or answer.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("context") or []
        contexts = []
        supporting = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, (list, tuple)) and len(paragraph) >= 2:
                title, text = paragraph[0], paragraph[1]
                is_supporting = False
            elif isinstance(paragraph, dict):
                title = _paragraph_title(paragraph, paragraph_index)
                text = _paragraph_text(paragraph)
                is_supporting = bool(
                    paragraph.get("is_supporting")
                    or paragraph.get("supporting")
                    or paragraph.get("is_support")
                )
            else:
                title = f"paragraph_{paragraph_index}"
                text = str(paragraph)
                is_supporting = False
            contexts.append({"title": str(title), "sentences": [str(text)]})
            if is_supporting:
                supporting.append([str(title), 0])
        examples.append(
            MusiqueExample(
                question_id=str(item.get("id") or item.get("_id") or item.get("question_id") or f"musique_{item_index:04d}"),
                question=str(item.get("question") or ""),
                answer=str(answer or item.get("gold_answer") or ""),
                contexts=contexts,
                supporting_facts=item.get("supporting_facts") or supporting,
            )
        )
    return examples


def select_pilot_examples(examples: Iterable[MusiqueExample], n: int) -> List[MusiqueExample]:
    selected = []
    for example in examples:
        if not example.question or not example.answer:
            continue
        if len(example.answer.split()) > 8:
            continue
        if len(example.contexts) < 2:
            continue
        selected.append(example)
        if len(selected) >= n:
            break
    return selected


def save_examples(examples: List[MusiqueExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([example.to_dict() for example in examples], indent=2), encoding="utf-8")


def load_prepared_examples(path: Path) -> List[MusiqueExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [MusiqueExample(**item) for item in raw]


def write_context_corpus(examples: List[MusiqueExample], output_dir: Path) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_titles = set()
    manifest = []
    for example in examples:
        for context in example.contexts:
            title = context.get("title", "document")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(str(sentence) for sentence in context.get("sentences", []))
            filename = f"{_slug(title)}.txt"
            (output_dir / filename).write_text(text, encoding="utf-8")
            manifest.append({"title": title, "filename": filename, "text": text})
    return manifest
