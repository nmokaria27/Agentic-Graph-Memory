#!/usr/bin/env bash
# EXP-2: LongMemEval B1-0 smoke — 5 knowledge-update questions, oracle split,
# Fireworks lane (deepseek-v4-flash + qwen3-embedding-8b). Zero gpu02 usage.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
export LLM_BACKEND=vllm
export VLLM_BASE_URL=https://api.fireworks.ai/inference/v1
export VLLM_API_KEY="$FIREWORKS_API_KEY"
export LLM_DEFAULT_MODEL=accounts/fireworks/models/deepseek-v4-flash
export VLLM_MAX_MODEL_LEN=131072
export EMBEDDING_BASE_URL=https://api.fireworks.ai/inference/v1
export EMBEDDING_API_KEY="$FIREWORKS_API_KEY"
export EMBEDDING_MODEL=accounts/fireworks/models/qwen3-embedding-8b

LOG=evaluation/results/exp2b_lme_smoke.log
CACHE=evaluation/results/lme_kg_cache_fw_smoke_flash
mkdir -p "$CACHE"

{
  echo "[EXP2B] start $(date) model=$LLM_DEFAULT_MODEL embed=$EMBEDDING_MODEL"
  python -u evaluation/LongMemEval/run_eval.py \
    --question-type knowledge-update --max-questions 5 \
    --save-kg-dir "$CACHE" \
    --output evaluation/results/exp2b_lme_smoke.json
  echo "[EXP2B] scoring (substring only; judge later against same-lane model)..."
  python evaluation/LongMemEval/score_longmemeval.py \
    --cache-dir "$CACHE" \
    --output evaluation/results/exp2b_lme_smoke_scores.json
  echo "EXP2B_DONE $(date)"
} >> "$LOG" 2>&1
