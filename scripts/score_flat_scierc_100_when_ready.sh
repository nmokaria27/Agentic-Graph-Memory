#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs evaluation/results /tmp/mpl

LOG="logs/scierc_flat_100_score_when_ready.log"
exec >>"$LOG" 2>&1

echo "[flat-score] started $(date)"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export OPENAI_API_KEY="${OPENAI_API_KEY_BACKUP:-${OPENAI_API_KEY:-}}"
export LLM_BACKEND=openai
export OPENAI_REASONING_EFFORT=minimal
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

BUILD_OUTPUT="evaluation/results/scierc_ungoverned_gpt5_test_100.json"
CANON_OUTPUT="evaluation/results/scierc_ungoverned_gpt5_test_100_canon.json"
SCORE_OUTPUT="evaluation/results/score_scierc_ungoverned_gpt5_test_100_canon.json"

while [[ ! -s "$BUILD_OUTPUT" ]]; do
  if pgrep -f "scripts/build_ungoverned_scierc.py --split test --max-docs 100" >/dev/null; then
    echo "[flat-score] active flat build still running; waiting"
    sleep 300
    continue
  fi

  echo "[flat-score] no active flat build and output missing; resuming flat build"
  export LLM_USAGE_LOG="evaluation/results/gpt5_scierc_ungov_100_usage.jsonl"
  ./.venv/bin/python -u scripts/build_ungoverned_scierc.py \
    --split test \
    --max-docs 100 \
    --model gpt-5 \
    --fixed-schema \
    --skip-evidence-linking \
    --skip-verification \
    --output "$BUILD_OUTPUT" \
    --stats-output evaluation/results/scierc_ungoverned_gpt5_test_100_stats.json \
    --checkpoint-every 1 \
    --resume-from-checkpoint
done

echo "[flat-score] build output found; canonicalizing and scoring"
./.venv/bin/python scripts/canonicalize_scierc_relations.py \
  --input "$BUILD_OUTPUT" \
  --output "$CANON_OUTPUT"
./.venv/bin/python scripts/score_accumulated_scierc_rich.py \
  --kg-path "$CANON_OUTPUT" \
  --split test \
  --max-docs 100 \
  --output "$SCORE_OUTPUT"
echo "[flat-score] complete $(date)"
