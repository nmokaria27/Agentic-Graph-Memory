#!/usr/bin/env bash
# EXP-SPGOV-2 (GB-9): governed singlepass re-run after GB-11 type propagation
# (commit 8ecd19b). Same slices and bars as EXP-SPGOV; fresh cache dir.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
# No overrides: .env points at gpu02 vLLM (Qwen3) + gpu01 embeddings.

LOG=evaluation/results/exp_spgov2.log
CACHE=evaluation/results/docred_kg_cache_spgov2
mkdir -p "$CACHE"

{
  echo "[SPGOV2] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  echo "[SPGOV2] === primary slice: docs 100-119 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 20 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov2_100_119.json
  echo "[SPGOV2] === diagnostic slice A: docs 0-4 ==="
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 0 --max-docs 5 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_spgov2_sliceA.json
  echo "[SPGOV2] scoring both slices..."
  python evaluation/DocRED/score_docred.py --kg-dir "$CACHE" --strategy spgov \
    --output evaluation/results/exp_spgov2_scores_all.json
  echo "SPGOV2_DONE $(date)"
} >> "$LOG" 2>&1
