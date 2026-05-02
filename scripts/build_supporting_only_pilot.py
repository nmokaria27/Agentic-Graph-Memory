"""Build a supporting-only variant of a MuSiQue pilot for fair RAG/GraphRAG re-runs.

Input: a prepared MuSiQue pilot JSON (e.g. musique_pilot_20.json) where each example
carries `contexts` with all 20 paragraphs and `supporting_facts` listing the gold-supporting titles.

Output: a pilot JSON where each example's `contexts` is restricted to its supporting paragraphs,
plus a docs directory (GraphRAG-style: one .txt per supporting paragraph title).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug or "document"


def _supporting_titles(supporting_facts: List[Any]) -> Set[str]:
    titles: Set[str] = set()
    for fact in supporting_facts or []:
        if isinstance(fact, (list, tuple)) and fact:
            titles.add(str(fact[0]))
        elif isinstance(fact, dict):
            t = fact.get("title") or fact.get("paragraph_title")
            if t:
                titles.add(str(t))
        elif isinstance(fact, str):
            titles.add(fact)
    return titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-docs-dir", required=True)
    args = parser.parse_args()

    examples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out_dir = Path(args.output_docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_examples: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()
    total_kept = 0
    for ex in examples:
        sup = _supporting_titles(ex.get("supporting_facts", []))
        kept = [c for c in ex.get("contexts", []) if str(c.get("title")) in sup]
        if not kept:
            print(f"WARN no supporting context kept for {ex.get('question_id')}")
            continue
        total_kept += len(kept)
        for c in kept:
            title = str(c.get("title"))
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(str(s) for s in c.get("sentences", []))
            (out_dir / f"{_slug(title)}.txt").write_text(text, encoding="utf-8")
        new_ex = dict(ex)
        new_ex["contexts"] = kept
        new_examples.append(new_ex)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(new_examples, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}: {len(new_examples)} examples, avg {total_kept/len(new_examples):.1f} ctx/ex")
    print(f"wrote docs dir {out_dir}: {len(seen_titles)} unique supporting paragraphs")


if __name__ == "__main__":
    main()
