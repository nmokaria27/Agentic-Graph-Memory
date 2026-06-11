# Project Handoff & Session Summary

## 1. The "Thinking Model vs. JSON Constraint" Conflict (Ollama)
*   **The Issue:** When using reasoning models like `gemma4:31b` or `deepseek-r1`, Ollama's strict `json_object` mode (GBNF grammar) blocks the model from outputting its `<think>...</think>` tags. Because the model *must* reason before answering, this constraint caused it to silently fail and return an empty string.
*   **The Fix (Implemented):** The `openai_client.py` retry logic was modified. Attempt 1 still tries strict JSON mode. If it returns an empty string (characteristic of the thinking conflict), Attempt 2 and 3 automatically drop the `response_format` constraint, allowing the model to "think" in free text. The `_extract_json` regex then surgically extracts the `{...}` payload from the response.

## 2. Dynamic Model Tiering Bug
*   **The Issue:** The `--model` CLI argument was being ignored by worker agents because `DEFAULT_MODEL_TIERS` in `BaseAgent` was evaluated at script import time, capturing the `os.getenv` values before `run_pipeline.py` could set them.
*   **The Fix (Implemented):** Refactored `BaseAgent` to dynamically evaluate model tiers via `get_default_model_tiers()` at runtime during the `call_llm` invocation.

## 3. The "Serial Batching Explosion" Bottleneck (4-Hour Stall)
*   **The Issue:** The pipeline appeared to "freeze" for over 4 hours with 87% GPU utilization. The root cause was that agents (`EntityExtractor` coreference, `RelationExtractor`, `EvidenceLinker`, etc.) were processing the ~2,500 entities in tiny, hardcoded batch sizes of 10 to 30. This generated over 120 sequential LLM calls for a single stage. With `gemma4:31b` taking ~2 mins per call, the pipeline hit a massive scaling bottleneck.
*   **The Fix (Implemented):** Systematically increased the `BATCH` sizes to **100** across all 5 major agent logic loops. Added verbose terminal logging to track "Batch X of Y" so the user is never left in the dark during long reasoning phases.

## 4. Optimal Multi-Model Configuration (The "Dream Team")
To avoid crippling latency while maintaining high extraction accuracy, the pipeline should use a tiered approach rather than forcing a 31B reasoning model to do simple parsing.
*   **SMALL Tier (Stages 1-2):** `qwen3:4b` (Ultra-fast document segmentation and domain classification).
*   **MEDIUM Tier (Stages 3-4):** `gemma3:27b` (Heavy extraction capable of following strict JSON schemas without the `<think>` tag overhead).
*   **LARGE Tier (Stages 5-9):** `gemma4:31b` (Deep reasoning reserved for evidence verification, fact-checking, and final deliberation).

**Run Command:**
```bash
export LLM_SMALL_MODEL="qwen3:4b"
export LLM_MEDIUM_MODEL="gemma3:27b"
export LLM_LARGE_MODEL="gemma4:31b"
python3 scripts/run_pipeline.py --input wikipedia_test.txt
```

## 5. GPU Server Routing (gpu01 vs. gpu02)
To run the pipeline on a different lab server (e.g., GPU02) without using `.env` files:
*   **If running via local SSH Tunnel:** Update your tunnel command to target the new host:
    ```bash
    ssh -L 11434:gpu02.mind.cs.umd.edu:11434 <your_id>@mind-access00.cs.umd.edu
    ```
    *(The Python script defaults to `localhost:11434` and will route through the tunnel automatically).*
*   **If running directly on the remote lab terminal (Jupyter/VS Code):** Export the base URL before running the script:
    ```bash
    export OLLAMA_BASE_URL="http://gpu02.mind.cs.umd.edu:11434/v1"
    python3 scripts/run_pipeline.py --input wikipedia_test.txt
    ```

## 6. Checkpoint & Resume (Re-added)
*   **Why it's back:** A 7-hour single-doc run (`wikipedia_test.txt`, 697 entities) survived stage 3 entity extraction but got stuck mid stage-4 RHF tail-binding with no way to recover the in-memory state (macOS SIP blocks pyrasite injection without sudo). The previous single-blob `pipeline_checkpoint.json` only fired between docs, not between stages, so it never would have helped.
*   **What it does now:** `multi_agent_kg/core/checkpoint.py` ships a `CheckpointManager` that writes a per-stage JSON file (`stage_<n>.json`) plus a full `governed_kg_latest.json` snapshot after every pipeline stage (1, 2, 2b, 3, 3b, 4, 4b, 5, 6, 8, 9). Writes are atomic via tmp + `os.replace`.
*   **Storage:** `checkpoints/{document_id}/manifest.json` indexes completed stages. Per-doc subdirs keep multi-doc corpora isolated. `checkpoints/` is gitignored.
*   **Usage:**
    ```bash
    # Fresh run (writes checkpoints automatically)
    python3 scripts/run_pipeline.py --input wikipedia_test.txt

    # Resume after a crash — skips completed stages
    python3 scripts/run_pipeline.py --input wikipedia_test.txt --resume

    # Disable entirely
    python3 scripts/run_pipeline.py --input wikipedia_test.txt --no-checkpoint
    ```
*   **Resume contract:** opt-in via `--resume`. The orchestrator loads the per-stage payload from disk and prints `SKIPPED (resumed)` for each stage that already has a checkpoint. The governed-KG snapshot from the most-advanced doc is restored at orchestrator construction time. For multi-doc corpora, each subsequent partially-completed doc rebinds the orchestrator + agents to *its own* per-doc `governed_kg_latest.json` via `_rebind_governed_kg()` so per-doc governance state survives the kill.
*   **Known resume limitations:**
    *   `SharedMemory` (entity aliases, blackboard hypotheses, working memory) is NOT checkpointed. Skipped stages don't re-register their side effects. Stage 6 deliberation and stage 9 knowledge organization rely mostly on the explicit `entities` / `triples` lists which ARE restored, so final KG quality should be unaffected — but deliberation voting metrics (`voting_sessions`, `debates_triggered`) for a resumed mid-doc run may differ from a fresh run.
    *   `pipeline_debug.log` is truncated by `DebugLogger(clear_log=True)` at every script invocation (including `--help`). Don't run `scripts/run_pipeline.py` while another instance owns the log file.
    *   Manifest is updated *after* the stage file is written. A SIGKILL between the two leaves an orphan stage file. `CheckpointManager.has()` reads the filesystem directly so resume still works; only `discover_checkpoints()` (which reads the manifest) under-reports.
*   **`recover_from_log.py`:** kept as an emergency fallback for runs that pre-date this rewrite. Not needed for new runs.