"""Prepare a small MuSiQue pilot corpus for governed KG QA experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.musique.utils import (
    load_musique_json,
    save_examples,
    select_pilot_examples,
    write_context_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, help="Raw MuSiQue JSON or JSONL file")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", default="evaluation/results/musique_pilot_20.json")
    parser.add_argument("--docs-dir", default="evaluation/results/musique_pilot_20_docs")
    parser.add_argument("--manifest", default="evaluation/results/musique_pilot_20_docs_manifest.json")
    args = parser.parse_args()

    examples = select_pilot_examples(load_musique_json(Path(args.source_json)), args.n)
    save_examples(examples, Path(args.output))
    manifest = write_context_corpus(examples, Path(args.docs_dir))
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "examples": len(examples),
        "unique_documents": len(manifest),
        "output": args.output,
        "docs_dir": args.docs_dir,
        "manifest": args.manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
