#!/usr/bin/env bash
# EXP-FRESHNESS-E2E (GB-2b), local leg: same 5 knowledge-update dev questions on
# gpu02 Qwen3-30B-A3B (production lane) + gpu01 Ollama embeddings — this is also
# the first honest B1-0 gate attempt on fully-fixed code + the production model.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
# No overrides: .env already points at gpu02 vLLM (Qwen3) + gpu01 embeddings.

LOG=evaluation/results/exp_freshness_e2e_local.log
CACHE=evaluation/results/lme_kg_cache_qwen3_freshness
mkdir -p "$CACHE"

{
  echo "[EXPFR-LOCAL] start $(date) model=$LLM_DEFAULT_MODEL embed=${EMBEDDING_MODEL:-mxbai}"
  python -u evaluation/LongMemEval/run_eval.py \
    --question-type knowledge-update --max-questions 5 \
    --save-kg-dir "$CACHE" \
    --output evaluation/results/exp_freshness_e2e_local.json
  echo "[EXPFR-LOCAL] scoring (substring only; judge later)..."
  python evaluation/LongMemEval/score_longmemeval.py \
    --cache-dir "$CACHE" \
    --output evaluation/results/exp_freshness_e2e_local_scores.json
  echo "EXPFR_LOCAL_DONE $(date)"
} >> "$LOG" 2>&1
