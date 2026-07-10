#!/usr/bin/env bash
# EXP-LITERAL-GUARD (GB-12) diagnostic: spgov docs 100-104, n=5 runs/doesn't
# smoke after the literal guard (commit 0015386). Fresh cache.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
LOG=evaluation/results/exp_literal_diag.log
CACHE=evaluation/results/docred_kg_cache_litguard
mkdir -p "$CACHE"
{
  echo "[LITGUARD] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  python -u evaluation/DocRED/run_eval.py --strategy spgov \
    --offset 100 --max-docs 5 \
    --save-kg-dir "$CACHE" --output evaluation/results/exp_literal_diag.json
  python evaluation/DocRED/score_docred.py --kg-dir "$CACHE" --strategy spgov \
    --output evaluation/results/exp_literal_diag_scores.json
  echo "LITGUARD_DONE $(date)"
} >> "$LOG" 2>&1
