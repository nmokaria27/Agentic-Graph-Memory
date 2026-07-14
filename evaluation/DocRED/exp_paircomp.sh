#!/usr/bin/env bash
# EXP-PAIR-COMPLETE (GB-3): spgov + stage-4c pair completion, docs 100-119,
# fan-out ON. Control = EXP-SPGOV-3 cache (same code, flag off).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export LLM_BATCH_CONCURRENCY=4
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
LOG=evaluation/results/exp_paircomp.log
CACHE=evaluation/results/docred_kg_cache_paircomp
mkdir -p "$CACHE"
{
  echo "[PAIRCOMP] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  python -u evaluation/DocRED/run_eval.py --strategy spgov --pair-completion \
    --offset 100 --max-docs 20 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_paircomp_100_119.json
  python evaluation/DocRED/score_docred.py --kg-dir "$CACHE" --strategy spgov \
    --output evaluation/results/exp_paircomp_scores.json
  echo "PAIRCOMP_DONE $(date)"
} >> "$LOG" 2>&1
