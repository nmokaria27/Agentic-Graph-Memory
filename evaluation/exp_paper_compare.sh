#!/usr/bin/env bash
# EXP-PAPER-COMPARE: paper Table 2 protocol (SciERC test 100 docs, fixed
# schema) on current HEAD + local Qwen3. Governed (MAGG) then flat insertion,
# canonicalize, score strict+mapped. Checkpointed per doc — rerun-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PYTHONPATH=.

LOG=evaluation/results/exp_paper_compare.log
GOV=evaluation/results/scierc_governed_qwen3_test_100.json
FLAT=evaluation/results/scierc_flat_qwen3_test_100.json

{
  echo "[PAPERCMP] start $(date) model=$LLM_DEFAULT_MODEL commit=$(git rev-parse --short HEAD)"

  echo "[PAPERCMP] === condition 1/2: MAGG governed build ==="
  python -u scripts/build_governed_scierc.py \
    --split test --max-docs 100 --model "$LLM_DEFAULT_MODEL" --fixed-schema \
    --checkpoint-every 1 --resume-from-checkpoint \
    --output "$GOV" \
    --stats-output evaluation/results/scierc_governed_qwen3_test_100_stats.json

  echo "[PAPERCMP] === condition 2/2: flat insertion build ==="
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
