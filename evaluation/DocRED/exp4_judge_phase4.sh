#!/usr/bin/env bash
# EXP-4: LLM-judge adjudication of Phase-4 relation quality (local lane, gpu02
# idle post-Phase-4). Scoring-only on cached predictions; judge verdicts cached
# per doc inside the kg-dir, so reruns are free.
set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
export NEMOTRON_THINKING=on

LOG=evaluation/results/exp4_judge.log
CACHE=evaluation/results/docred_kg_cache_phase4

{
  echo "[EXP4] start $(date) judge=$LLM_DEFAULT_MODEL"
  for strat in singlepass hybrid; do
    echo "[EXP4] === judging strategy=$strat ==="
    python -u evaluation/DocRED/score_docred.py \
      --kg-dir "$CACHE" --strategy "$strat" --judge \
      --output "evaluation/results/exp4_judge_scores_${strat}.json"
  done
  echo "EXP4_DONE $(date)"
} >> "$LOG" 2>&1
