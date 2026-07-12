#!/usr/bin/env bash
# EXP-ASYNC-BATCH (GB-4) Fireworks A/B: spgov docs 100-104, deepseek-v4-flash,
# arm A concurrency=1 (control) vs arm B concurrency=4. Run FROM THE WORKTREE
# (main tree frozen by EXP-PAPER-COMPARE). Wall + counts only, no scoring.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source /home/nmokaria/Agentic-Graph-Memory/.env; set +a
export LLM_BACKEND=vllm
export VLLM_BASE_URL=https://api.fireworks.ai/inference/v1
export VLLM_API_KEY="$FIREWORKS_API_KEY"
export LLM_DEFAULT_MODEL=accounts/fireworks/models/deepseek-v4-flash
export PYTHONPATH=.

LOG=evaluation/results/exp_async_batch_fw.log
mkdir -p evaluation/results

{
  echo "[ASYNCB] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  echo "[ASYNCB] === arm A: LLM_BATCH_CONCURRENCY=1 (control) ==="
  LLM_BATCH_CONCURRENCY=1 python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 5 \
    --save-kg-dir evaluation/results/docred_kg_cache_fw_asyncA \
    --output evaluation/results/exp_async_batch_armA.json
  echo "[ASYNCB] === arm B: LLM_BATCH_CONCURRENCY=4 ==="
  LLM_BATCH_CONCURRENCY=4 python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 5 \
    --save-kg-dir evaluation/results/docred_kg_cache_fw_asyncB \
    --output evaluation/results/exp_async_batch_armB.json
  echo "ASYNCB_DONE $(date)"
} >> "$LOG" 2>&1
