#!/usr/bin/env bash
# EXP-MODEL-LOCAL-2: swap gpu02 to gpt-oss-120b and run the qualification gate.
# Restores Qwen3 automatically on serving failure.
set -uo pipefail
cd /home/nmokaria/Agentic-Graph-Memory
set -a; source .env; set +a
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
export PYTHONPATH=.
LOG=evaluation/results/exp_model_local2.log
GPTOSS_MODEL="openai/gpt-oss-120b"

restore_qwen3() {
  echo "[ML2] RESTORING QWEN3 $(date)"
  pkill -f "vllm.entrypoints.openai.api_server" || true; sleep 8
  nohup env TRITON_CACHE_DIR=/scratch/triton_cache_qwen3 VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_VISIBLE_DEVICES=0,1 /home/nmokaria/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /scratch/models/qwen3-30b-a3b-instruct-2507-fp8 \
    --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 --port 8000 --host 0.0.0.0 \
    --max-model-len 131072 --gpu-memory-utilization 0.85 \
    > /scratch/vllm_qwen3.log 2>&1 &
}

{
  echo "[ML2] start $(date) commit=$(git rev-parse --short HEAD)"
  echo "[ML2] === (a) swap to gpt-oss-120b ==="
  pkill -f "vllm.entrypoints.openai.api_server" || true; sleep 8
  nohup env TRITON_CACHE_DIR=/scratch/triton_cache_gptoss VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_VISIBLE_DEVICES=0,1 /home/nmokaria/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /scratch/models/gpt-oss-120b \
    --served-model-name "$GPTOSS_MODEL" \
    --tensor-parallel-size 2 --port 8000 --host 0.0.0.0 \
    --max-model-len 32768 --gpu-memory-utilization 0.92 \
    > /scratch/vllm_gptoss.log 2>&1 &
  for i in $(seq 1 120); do
    sleep 15
    curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8000/health 2>/dev/null | grep -q 200 && break
    if ! pgrep -f "vllm.entrypoints" >/dev/null; then
      echo "[ML2] GATE (a) FAIL: vllm process died during load (see /scratch/vllm_gptoss.log)"
      restore_qwen3; echo "ML2_DONE_FAIL_A $(date)"; exit 1
    fi
    [ "$i" = 120 ] && { echo "[ML2] GATE (a) FAIL: no health after 30 min"; restore_qwen3; echo "ML2_DONE_FAIL_A $(date)"; exit 1; }
  done
  curl -s localhost:8000/v1/models | grep -q "gpt-oss-120b" || { echo "[ML2] GATE (a) FAIL: wrong model name"; restore_qwen3; echo "ML2_DONE_FAIL_A $(date)"; exit 1; }
  echo "[ML2] gate (a) PASS $(date)"

  echo "[ML2] === (b) JSON-shape smoke through project client ==="
  LLM_DEFAULT_MODEL="$GPTOSS_MODEL" python -c "
from multi_agent_kg.llm.openai_client import chat_completion_json
r = chat_completion_json(prompt='Extract entities from: Marie Curie won the Nobel Prize in Paris. Return {\"entities\":[{\"text\":...,\"type\":...}]}', system_prompt='Return only JSON.', max_tokens=1024)
assert isinstance(r, dict) and r.get('entities'), f'bad shape: {r!r}'
print('[ML2] gate (b) PASS —', len(r['entities']), 'entities')
" || { echo "[ML2] GATE (b) FAIL"; restore_qwen3; echo "ML2_DONE_FAIL_B $(date)"; exit 1; }

  echo "[ML2] === (c) slice A singlepass ==="
  LLM_DEFAULT_MODEL="$GPTOSS_MODEL" python -u evaluation/DocRED/run_eval.py --strategy singlepass \
    --offset 0 --max-docs 5 \
    --save-kg-dir evaluation/results/docred_kg_cache_gptoss_sliceA \
    --output evaluation/results/ml2_sliceA.json
  python evaluation/DocRED/score_docred.py --kg-dir evaluation/results/docred_kg_cache_gptoss_sliceA \
    --strategy singlepass --output evaluation/results/ml2_sliceA_scores.json

  echo "[ML2] === (d) SciERC 10-doc governed rebuild ==="
  LLM_DEFAULT_MODEL="$GPTOSS_MODEL" python -u scripts/build_governed_scierc.py \
    --split test --max-docs 10 --model "$GPTOSS_MODEL" --fixed-schema \
    --checkpoint-every 1 --resume-from-checkpoint \
    --output evaluation/results/scierc_governed_gptoss_local_10.json \
    --stats-output evaluation/results/scierc_governed_gptoss_local_10_stats.json
  python scripts/canonicalize_scierc_relations.py --input evaluation/results/scierc_governed_gptoss_local_10.json \
    --output evaluation/results/scierc_governed_gptoss_local_10_canon.json
  python scripts/score_accumulated_scierc_rich.py \
    --kg-path evaluation/results/scierc_governed_gptoss_local_10_canon.json \
    --split test --max-docs 10 \
    --output evaluation/results/scierc_governed_gptoss_local_10_scores.json
  echo "ML2_DONE $(date)"
} >> "$LOG" 2>&1
