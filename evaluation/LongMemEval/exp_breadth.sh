#!/usr/bin/env bash
# EXP-LME-BREADTH: 8 dev questions x 6 ability types on merged HEAD + Qwen3.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export LLM_BATCH_CONCURRENCY=4
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
LOG=evaluation/results/exp_lme_breadth.log
{
  echo "[LMEB] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"
  curl -s gpu02.mind.cs.umd.edu:8000/v1/models | grep -o 'Qwen[^"]*' | head -1 || { echo "[LMEB] PREFLIGHT FAIL: wrong/absent model"; exit 1; }
  for QT in knowledge-update temporal-reasoning multi-session single-session-user single-session-assistant single-session-preference; do
    echo "[LMEB] === $QT === $(date)"
    python -u evaluation/LongMemEval/run_eval.py --question-type "$QT" \
      --max-questions 8 \
      --save-kg-dir "evaluation/results/lme_breadth_$QT" \
      --output "evaluation/results/lme_breadth_$QT.json" || echo "[LMEB] WARNING: $QT leg exited nonzero — continuing"
  done
  echo "LMEB_DONE $(date)"
} >> "$LOG" 2>&1
