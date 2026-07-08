#!/usr/bin/env bash
# EXP-FRESHNESS-E2E (GB-2b), Fireworks leg: 5 knowledge-update dev questions on
# deepseek-v4-flash — apples-to-apples against lme_kg_cache_fw_validate (same
# model/questions, pre-GB-2b code). Zero gpu02 usage.
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

LOG=evaluation/results/exp_freshness_e2e_fw.log
CACHE=evaluation/results/lme_kg_cache_fw_freshness
mkdir -p "$CACHE"

{
  echo "[EXPFR-FW] start $(date) model=$LLM_DEFAULT_MODEL embed=$EMBEDDING_MODEL"
  python -u evaluation/LongMemEval/run_eval.py \
    --question-type knowledge-update --max-questions 5 \
    --save-kg-dir "$CACHE" \
    --output evaluation/results/exp_freshness_e2e_fw.json
  echo "[EXPFR-FW] scoring (substring only; judge later)..."
  python evaluation/LongMemEval/score_longmemeval.py \
    --cache-dir "$CACHE" \
    --output evaluation/results/exp_freshness_e2e_fw_scores.json
  echo "EXPFR_FW_DONE $(date)"
} >> "$LOG" 2>&1
