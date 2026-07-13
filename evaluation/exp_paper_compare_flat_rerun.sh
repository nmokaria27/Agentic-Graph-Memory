#!/usr/bin/env bash
# EXP-PAPER-COMPARE flat leg re-run on FIXED code (ungoverned alias-sync
# guard + embedding failover). Embeddings: gpu02-local Ollama primary,
# Fireworks auto-failover armed via .env. Scoring: the paper's own tools
# (canonicalize -> score_accumulated_scierc_rich), both legs.
set -uo pipefail
cd /home/nmokaria/Agentic-Graph-Memory
set -a; source .env; set +a
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
export PYTHONPATH=.
LOG=evaluation/results/exp_paper_compare.log
FLAT=evaluation/results/scierc_flat_qwen3_test_100.json
{
  echo "[PAPERCMP] flat-leg RERUN on fixed code $(date) commit=$(git rev-parse --short HEAD)"
  python -u scripts/build_ungoverned_scierc.py \
    --split test --max-docs 100 --model "$LLM_DEFAULT_MODEL" --fixed-schema \
    --skip-evidence-linking --skip-verification \
    --checkpoint-every 1 --resume-from-checkpoint \
    --output "$FLAT" \
    --stats-output evaluation/results/scierc_flat_qwen3_test_100_stats.json
  echo "[PAPERCMP] === canonicalize + rich scoring (paper protocol) ==="
  python scripts/canonicalize_scierc_relations.py --input "$FLAT" \
    --output evaluation/results/scierc_flat_qwen3_test_100_canon.json
  python scripts/score_accumulated_scierc_rich.py \
    --kg-path evaluation/results/scierc_flat_qwen3_test_100_canon.json \
    --split test --max-docs 100 \
    --output evaluation/results/scierc_flat_qwen3_test_100_rich_scores.json
  echo "PAPERCMP_FLAT_DONE $(date)"
} >> "$LOG" 2>&1
