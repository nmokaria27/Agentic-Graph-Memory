"""Prepare a small HotpotQA distractor pilot for overnight QA experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import (
    load_hotpot_json,
    save_examples,
    select_pilot_examples,
    write_context_corpus,
)


DEFAULT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", default="evaluation/results/hotpotqa_dev_distractor_v1.json")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", default="evaluation/results/hotpotqa_pilot_20.json")
    parser.add_argument("--docs-dir", default="evaluation/results/hotpotqa_pilot_20_docs")
    parser.add_argument("--manifest", default="evaluation/results/hotpotqa_pilot_20_docs_manifest.json")
    args = parser.parse_args()

    source = Path(args.source_json)
    if not source.exists():
        _download(args.url, source)

    examples = load_hotpot_json(source)
    selected = select_pilot_examples(examples, args.n)
    if len(selected) < args.n:
        raise SystemExit(f"Only selected {len(selected)} examples; requested {args.n}")

    output = Path(args.output)
    docs_dir = Path(args.docs_dir)
    manifest_path = Path(args.manifest)
    save_examples(selected, output)
    manifest = write_context_corpus(selected, docs_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "examples": len(selected),
                "unique_context_documents": len(manifest),
                "output": str(output),
                "docs_dir": str(docs_dir),
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
