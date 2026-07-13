#!/usr/bin/env bash
# EXP-SPGOV-2 (GB-9): governed singlepass re-run after GB-11 type propagation
# (commit 8ecd19b). Same slices and bars as EXP-SPGOV; fresh cache dir.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
export LLM_BATCH_CONCURRENCY=4
# No overrides: .env points at gpu02 vLLM (Qwen3) + gpu01 embeddings.

LOG=evaluation/results/exp_spgov3.log
CACHE=evaluation/results/docred_kg_cache_spgov3
mkdir -p "$CACHE"

{
  echo "[SPGOV3] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  echo "[SPGOV3] === primary slice: docs 100-119 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 20 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov3_100_119.json
  echo "[SPGOV3] === diagnostic slice A: docs 0-4 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 0 --max-docs 5 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov3_sliceA.json
  echo "[SPGOV3] scoring both slices..."
  python evaluation/DocRED/score_docred.py --kg-dir "$CACHE" --strategy spgov \
    --output evaluation/results/exp_spgov3_scores_all.json
  echo "SPGOV3_DONE $(date)"
} >> "$LOG" 2>&1
