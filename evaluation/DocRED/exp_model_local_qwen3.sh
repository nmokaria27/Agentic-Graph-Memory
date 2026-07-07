#!/usr/bin/env bash
# EXP-MODEL-LOCAL: slice A (docs 0-4) singlepass + hybrid on the local Qwen3
# vLLM (gpu02). Embeddings stay on gpu01 so extraction model is the only
# variable vs the Nemotron caches. Fresh cache dir (never mix models).
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
# .env already points LLM_DEFAULT_MODEL at Qwen/Qwen3-30B-A3B-Instruct-2507
# and VLLM_BASE_URL at the local server.

LOG=evaluation/results/exp_model_local_qwen3.log
CACHE=evaluation/results/docred_kg_cache_qwen3
mkdir -p "$CACHE"

{
  echo "[EXPML] start $(date) model=$LLM_DEFAULT_MODEL base=$VLLM_BASE_URL"
  for strat in singlepass hybrid; do
    echo "[EXPML] === strategy=$strat ==="
    python -u evaluation/DocRED/run_eval.py \
      --strategy "$strat" --max-docs 5 --offset 0 \
      --save-kg-dir "$CACHE" \
      --output "evaluation/results/expml_qwen3_${strat}.json"
  done
  echo "[EXPML] scoring..."
  for strat in singlepass hybrid; do
    python evaluation/DocRED/score_docred.py \
      --kg-dir "$CACHE" --strategy "$strat" \
      --output "evaluation/results/expml_qwen3_scores_${strat}.json"
  done
  echo "EXPML_DONE $(date)"
} >> "$LOG" 2>&1
