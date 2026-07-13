#!/usr/bin/env bash
# EXP-PAPER-COMPARE flat leg resume: gpu01 Ollama wedged mid-experiment;
# embeddings re-served from gpu02-local Ollama (SAME model+weights,
# mxbai-embed-large:335m, CPU) — substitution logged in EXPERIMENT_LOG.
# Runs against the MAIN tree (cd below), like the original script.
set -uo pipefail
cd /home/nmokaria/Agentic-Graph-Memory
set -a; source .env; set +a
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
export PYTHONPATH=.
LOG=evaluation/results/exp_paper_compare.log
FLAT=evaluation/results/scierc_flat_qwen3_test_100.json
{
  echo "[PAPERCMP] flat-leg RESUME $(date) — embeddings moved to gpu02-local ollama (same model) after gpu01 wedge"
  python -u scripts/build_ungoverned_scierc.py \
    --split test --max-docs 100 --model "$LLM_DEFAULT_MODEL" --fixed-schema \
    --skip-evidence-linking --skip-verification \
    --checkpoint-every 1 --resume-from-checkpoint \
    --output "$FLAT" \
    --stats-output evaluation/results/scierc_flat_qwen3_test_100_stats.json
  echo "[PAPERCMP] === canonicalize + score ==="
  for kind in governed flat; do
    src=evaluation/results/scierc_${kind}_qwen3_test_100.json
    canon=evaluation/results/scierc_${kind}_qwen3_test_100_canon.json
    python scripts/canonicalize_scierc_relations.py --input "$src" --output "$canon"
    python evaluation/evaluate_kg.py --gold evaluation/datasets/scierc/test.json \
      --predicted "$src" \
      --output-json evaluation/results/scierc_${kind}_qwen3_test_100_strict_scores.json
    python evaluation/evaluate_kg.py --gold evaluation/datasets/scierc/test.json \
      --predicted "$canon" \
      --output-json evaluation/results/scierc_${kind}_qwen3_test_100_mapped_scores.json
  done
  echo "PAPERCMP_DONE $(date)"
} >> "$LOG" 2>&1
