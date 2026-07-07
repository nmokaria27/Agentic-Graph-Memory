#!/usr/bin/env bash
# EXP-1: model-headroom baseline — slice A (docs 0-4) on Fireworks glm-5p2.
# Pure env-override model swap; no code changes; zero gpu02 usage.
# Fresh cache dir per SKILL.md rule (never mix models in one cache).
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
export LLM_BACKEND=vllm
export VLLM_BASE_URL=https://api.fireworks.ai/inference/v1
export VLLM_API_KEY="$FIREWORKS_API_KEY"
export LLM_DEFAULT_MODEL=accounts/fireworks/models/glm-5p2
export VLLM_MAX_MODEL_LEN=131072

LOG=evaluation/results/fw1_glm_sliceA.log
CACHE=evaluation/results/docred_kg_cache_fw_glm
mkdir -p "$CACHE"

{
  echo "[FW1] start $(date) model=$LLM_DEFAULT_MODEL"
  for strat in singlepass hybrid; do
    echo "[FW1] === strategy=$strat ==="
    python -u evaluation/DocRED/run_eval.py \
      --strategy "$strat" --max-docs 5 --offset 0 \
      --save-kg-dir "$CACHE" \
      --output "evaluation/results/fw1_glm_${strat}.json"
  done
  echo "[FW1] scoring..."
  for strat in singlepass hybrid; do
    python evaluation/DocRED/score_docred.py \
      --kg-dir "$CACHE" --strategy "$strat" \
      --output "evaluation/results/fw1_glm_scores_${strat}.json"
  done
  echo "FW1_DONE $(date)"
} >> "$LOG" 2>&1
