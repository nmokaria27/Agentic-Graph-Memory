#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OPENAI_API_KEY="${OPENAI_API_KEY_BACKUP:?OPENAI_API_KEY_BACKUP must be set}"
export LLM_BACKEND="${LLM_BACKEND:-openai}"
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-minimal}"
export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p logs /tmp/mpl

GOV_KG="evaluation/results/scierc_governed_gpt5_test_100_neighborhood.json"
GOV_CKPT="${GOV_KG}.checkpoint.json"
UNGOV_KG="evaluation/results/scierc_ungoverned_gpt5_test_100.json"
UNGOV_CANON="evaluation/results/scierc_ungoverned_gpt5_test_100_canon.json"

echo "[followup] waiting for governed 100-doc checkpoint..."
while true; do
  if ./.venv/bin/python -c "import json,sys,os; p='$GOV_CKPT'; d=json.load(open(p)) if os.path.exists(p) else {}; sys.exit(0 if len(d.get('_processed_doc_ids', [])) >= 100 else 1)"; then
    break
  fi
  sleep 300
done

echo "[followup] governed checkpoint complete; extending flat GPT-5 to 100 docs"
export LLM_USAGE_LOG=evaluation/results/gpt5_scierc_ungov_100_usage.jsonl
./.venv/bin/python -u scripts/build_ungoverned_scierc.py \
  --split test \
  --max-docs 100 \
  --model gpt-5 \
  --fixed-schema \
  --skip-evidence-linking \
  --skip-verification \
  --output "$UNGOV_KG" \
  --stats-output evaluation/results/scierc_ungoverned_gpt5_test_100_stats.json \
  --checkpoint-every 1 \
  --resume-from-checkpoint \
  >> logs/scierc_gpt5_ungov_100.log 2>&1

echo "[followup] canonicalizing flat relation labels"
./.venv/bin/python scripts/canonicalize_scierc_relations.py \
  --input "$UNGOV_KG" \
  --output "$UNGOV_CANON"

echo "[followup] scoring flat and basic-governed 100-doc artifacts"
./.venv/bin/python scripts/score_accumulated_scierc_rich.py \
  --kg-path "$UNGOV_CANON" \
  --split test \
  --max-docs 100 \
  --output evaluation/results/score_scierc_ungoverned_gpt5_test_100_canon.json
./.venv/bin/python scripts/score_accumulated_scierc_rich.py \
  --kg-path "$GOV_KG" \
  --split test \
  --max-docs 100 \
  --output evaluation/results/score_scierc_governed_gpt5_test_100_neighborhood.json

echo "[followup] running global reviewer 100-doc ablation"
export LLM_USAGE_LOG=evaluation/results/gpt5_ablation_global_minimal_reviewer_100_neighborhood_usage.jsonl
./.venv/bin/python -u scripts/run_governance_replay_ablation.py \
  --governed "$GOV_KG" \
  --mode global_llm_reviewer \
  --review-prompt-mode minimal \
  --model gpt-5 \
  --batch-size 20 \
  --output evaluation/results/ablation_global_minimal_reviewer_gpt5_100_neighborhood.json \
  --decisions-output evaluation/results/ablation_global_minimal_reviewer_gpt5_100_neighborhood_decisions.json \
  --stats-output evaluation/results/ablation_global_minimal_reviewer_gpt5_100_neighborhood_stats.json \
  >> logs/ablation_global_minimal_reviewer_gpt5_100_neighborhood.log 2>&1
./.venv/bin/python scripts/score_accumulated_scierc_rich.py \
  --kg-path evaluation/results/ablation_global_minimal_reviewer_gpt5_100_neighborhood.json \
  --split test \
  --max-docs 100 \
  --output evaluation/results/score_ablation_global_minimal_reviewer_gpt5_100_neighborhood.json

echo "[followup] running domain-memory reviewer 100-doc ablation"
export LLM_USAGE_LOG=evaluation/results/gpt5_ablation_domain_memory_reviewer_100_neighborhood_usage.jsonl
./.venv/bin/python -u scripts/run_governance_replay_ablation.py \
  --governed "$GOV_KG" \
  --mode domain_memory_reviewer \
  --model gpt-5 \
  --batch-size 20 \
  --min-confidence 0.75 \
  --output evaluation/results/ablation_domain_memory_reviewer_gpt5_100_neighborhood.json \
  --decisions-output evaluation/results/ablation_domain_memory_reviewer_gpt5_100_neighborhood_decisions.json \
  --stats-output evaluation/results/ablation_domain_memory_reviewer_gpt5_100_neighborhood_stats.json \
  >> logs/ablation_domain_memory_reviewer_gpt5_100_neighborhood.log 2>&1
./.venv/bin/python scripts/score_accumulated_scierc_rich.py \
  --kg-path evaluation/results/ablation_domain_memory_reviewer_gpt5_100_neighborhood.json \
  --split test \
  --max-docs 100 \
  --output evaluation/results/score_ablation_domain_memory_reviewer_gpt5_100_neighborhood.json

echo "[followup] done"
