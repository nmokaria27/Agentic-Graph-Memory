#!/usr/bin/env python3
"""Summarize JSONL LLM usage logs for cost-analysis tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _summarize(path: str) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    prompt = sum(int(r.get("prompt_tokens", 0) or 0) for r in records)
    completion = sum(int(r.get("completion_tokens", 0) or 0) for r in records)
    cached = sum(int(r.get("cached_tokens", 0) or 0) for r in records)
    reasoning = sum(int(r.get("reasoning_tokens", 0) or 0) for r in records)
    models = sorted({str(r.get("model", "")) for r in records if r.get("model")})
    return {
        "path": path,
        "calls": len(records),
        "models": models,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "total_tokens": prompt + completion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    payload = {"logs": [_summarize(path) for path in args.logs]}
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
