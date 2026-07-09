#!/usr/bin/env bash
# EXP-SPGOV (GB-9): governed singlepass on local Qwen3 (gpu02) —
# primary slice docs 100-119 (n=20, vs cached singlepass baseline) +
# diagnostic slice A (docs 0-4, vs cached singlepass/hybrid).
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
# No overrides: .env points at gpu02 vLLM (Qwen3) + gpu01 embeddings.

LOG=evaluation/results/exp_spgov.log
CACHE=evaluation/results/docred_kg_cache_spgov
mkdir -p "$CACHE"

{
  echo "[SPGOV] start $(date) model=$LLM_DEFAULT_MODEL"
  echo "[SPGOV] === primary slice: docs 100-119 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 20 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov_100_119.json
  echo "[SPGOV] === diagnostic slice A: docs 0-4 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 0 --max-docs 5 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov_sliceA.json
  echo "[SPGOV] scoring both slices..."
  python evaluation/DocRED/score_docred.py --kg-dir "$CACHE" --strategy spgov \
    --output evaluation/results/exp_spgov_scores_all.json
  echo "SPGOV_DONE $(date)"
} >> "$LOG" 2>&1
