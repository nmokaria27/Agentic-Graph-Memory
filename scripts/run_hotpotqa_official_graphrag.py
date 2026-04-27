"""Run official Microsoft GraphRAG on a prepared HotpotQA pilot set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.hotpotqa.utils import (
    aggregate_answer_metrics,
    answer_metrics,
    clean_prediction_for_scoring,
    load_prepared_examples,
)


SETTINGS_TEMPLATE = """completion_models:
  default_completion_model:
    model_provider: ollama
    model: {completion_model}
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    api_base: {completion_api_base}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: ollama
    model: {embedding_model}
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    api_base: {embedding_api_base}
    retry:
      type: exponential_backoff

input:
  type: text

chunking:
  type: tokens
  size: 1200
  overlap: 100
  encoding_model: o200k_base

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [Person, Organization, Location, Event, Work, Concept, Other]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false
  completion_model_id: default_completion_model
  prompt: "prompts/extract_claims.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1

community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

snapshots:
  graphml: false
  embeddings: false

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
{workflow_override}
"""


def _run(cmd: List[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def _clean_graphrag_stdout(stdout: str) -> str:
    kept: List[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("INFO:", "SUCCESS:", "WARNING:", "ERROR:")):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _prepare_root(args: argparse.Namespace) -> None:
    root = Path(args.graphrag_root)
    input_dir = root / "input"
    root.mkdir(parents=True, exist_ok=True)
    if args.rebuild_root and root.exists():
        shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    if not (root / "settings.yaml").exists():
        init = _run(
            [
                args.graphrag_bin,
                "init",
                "--root",
                str(root),
                "--model",
                args.model,
                "--embedding",
                args.embedding_model,
                "--force",
            ],
            timeout=180,
        )
        if init.returncode != 0:
            raise SystemExit(f"graphrag init failed:\nSTDOUT:\n{init.stdout}\nSTDERR:\n{init.stderr}")

    input_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(args.docs_dir).glob("*.txt"):
        shutil.copy2(path, input_dir / path.name)

    (root / ".env").write_text("GRAPHRAG_API_KEY=ollama\n", encoding="utf-8")
    (root / "settings.yaml").write_text(
        SETTINGS_TEMPLATE.format(
            completion_model=args.model,
            completion_api_base=args.completion_api_base,
            embedding_model=args.embedding_model,
            embedding_api_base=args.embedding_api_base,
            workflow_override=_workflow_override(args),
        ),
        encoding="utf-8",
    )


def _workflow_override(args: argparse.Namespace) -> str:
    if not args.skip_community_reports:
        return ""
    if args.index_method == "standard":
        workflows = [
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "extract_covariates",
            "create_communities",
            "create_final_text_units",
            "generate_text_embeddings",
        ]
    else:
        workflows = [
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph_nlp",
            "prune_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "generate_text_embeddings",
        ]
    quoted = ", ".join(f'"{name}"' for name in workflows)
    return f"\nworkflows: [{quoted}]\n"


def _ensure_index(args: argparse.Namespace) -> None:
    output_dir = Path(args.graphrag_root) / "output"
    if output_dir.exists() and list(output_dir.rglob("*.parquet")) and not args.force_index:
        print(f"GraphRAG index already exists under {output_dir}", flush=True)
        return
    completed = _run(
        [
            args.graphrag_bin,
            "index",
            "--root",
            args.graphrag_root,
            "--method",
            args.index_method,
            "-v",
        ],
        timeout=args.index_timeout_seconds,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "graphrag index failed:\n"
            f"STDOUT tail:\n{completed.stdout[-8000:]}\n"
            f"STDERR tail:\n{completed.stderr[-8000:]}"
        )


def _query(root: str, question: str, args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    try:
        completed = _run(
            [
                args.graphrag_bin,
                "query",
                "--root",
                root,
                "--method",
                args.query_method,
                "--response-type",
                args.response_type,
                question,
            ],
            timeout=args.query_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "answer": "",
            "returncode": -1,
            "stderr": f"timeout after {args.query_timeout_seconds}s",
            "raw_stdout": (exc.stdout or "")[-4000:],
            "duration_seconds": round(time.time() - start, 3),
        }
    answer = _clean_graphrag_stdout(completed.stdout)
    if completed.returncode != 0:
        answer = ""
    return {
        "answer": answer,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4000:],
        "raw_stdout": completed.stdout[-4000:],
        "duration_seconds": round(time.time() - start, 3),
    }


def _load_existing(output: Path) -> List[Dict[str, Any]]:
    if not output.exists():
        return []
    return json.loads(output.read_text(encoding="utf-8")).get("results", [])


def _write_output(output: Path, *, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": "HotpotQA distractor dev pilot",
                "system": f"official_graphrag_{args.query_method}",
                "model": args.model,
                "graphrag_root": args.graphrag_root,
                "examples_json": args.examples_json,
                "aggregate_metrics": aggregate_answer_metrics(rows),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", default="evaluation/results/hotpotqa_pilot_20.json")
    parser.add_argument("--docs-dir", default="evaluation/results/hotpotqa_pilot_20_docs")
    parser.add_argument("--graphrag-root", default="evaluation/results/hotpotqa_graphrag_20")
    parser.add_argument("--graphrag-bin", default=shutil.which("graphrag") or "graphrag")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--embedding-model", default="mxbai-embed-large:latest")
    parser.add_argument("--completion-api-base", default="http://127.0.0.1:11435")
    parser.add_argument("--embedding-api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--index-method", default="standard", choices=["standard", "fast"])
    parser.add_argument("--query-method", default="local", choices=["local", "global", "drift", "basic"])
    parser.add_argument("--response-type", default="Single concise answer")
    parser.add_argument("--index-timeout-seconds", type=int, default=21600)
    parser.add_argument("--query-timeout-seconds", type=int, default=900)
    parser.add_argument("--output", default="evaluation/results/hotpotqa_official_graphrag_20.json")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--rebuild-root", action="store_true")
    parser.add_argument(
        "--skip-community-reports",
        action="store_true",
        help="Skip expensive GraphRAG community report generation; intended for basic-search baselines.",
    )
    args = parser.parse_args()

    _prepare_root(args)
    if not args.skip_index:
        _ensure_index(args)

    examples = load_prepared_examples(Path(args.examples_json))
    output = Path(args.output)
    rows = _load_existing(output)
    completed_ids = {row.get("question_id") for row in rows}

    for i, example in enumerate(examples, start=1):
        if example.question_id in completed_ids:
            print(f"[{i}/{len(examples)}] skip {example.question_id}", flush=True)
            continue
        print(f"[{i}/{len(examples)}] official GraphRAG {example.question_id}", flush=True)
        result = _query(args.graphrag_root, example.question, args)
        answer = result.get("answer", "")
        row = {
            "question_id": example.question_id,
            "question": example.question,
            "gold_answer": example.answer,
            "answer": answer,
            "scored_answer": clean_prediction_for_scoring(answer, example.answer),
            "metrics": answer_metrics(answer, example.answer),
            "retrieval_metadata": result,
        }
        rows.append(row)
        _write_output(output, args=args, rows=rows)
        print(
            f"[{i}/{len(examples)}] answer={answer[:120]!r} gold={example.answer!r} "
            f"em={row['metrics']['exact_match']:.1f} f1={row['metrics']['token_f1']:.3f}",
            flush=True,
        )

    _write_output(output, args=args, rows=rows)
    print(json.dumps(aggregate_answer_metrics(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
