#!/usr/bin/env bash
# EXP-QA-PATH (GB-15) validation: requery cached breadth KGs (copies) with the
# freshness-aware QA. Swaps Qwen3 back first (baseline model comparability).
set -uo pipefail
cd /home/nmokaria/Agentic-Graph-Memory
set -a; source .env; set +a
export EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
export PYTHONPATH=.
LOG=evaluation/results/exp_qa_path.log
{
  echo "[QAPATH] start $(date) commit=$(git rev-parse --short HEAD)"
  echo "[QAPATH] === swap back to Qwen3 ==="
  pkill -f "vllm.entrypoints.openai.api_server" || true; sleep 8
  nohup env TRITON_CACHE_DIR=/scratch/triton_cache_qwen3 VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_VISIBLE_DEVICES=0,1 /home/nmokaria/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /scratch/models/qwen3-30b-a3b-instruct-2507-fp8 \
    --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 --port 8000 --host 0.0.0.0 \
    --max-model-len 131072 --gpu-memory-utilization 0.85 \
    > /scratch/vllm_qwen3.log 2>&1 &
  for i in $(seq 1 80); do
    sleep 15
    curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8000/health 2>/dev/null | grep -q 200 && break
    [ "$i" = 80 ] && { echo "[QAPATH] FAIL: qwen3 no health"; echo "QAPATH_DONE_FAIL $(date)"; exit 1; }
  done
  curl -s -H "Authorization: Bearer $VLLM_API_KEY" localhost:8000/v1/models | grep -q "Qwen3" || { echo "[QAPATH] FAIL: wrong model"; echo "QAPATH_DONE_FAIL $(date)"; exit 1; }
  echo "[QAPATH] qwen3 healthy $(date)"

  for QT in knowledge-update temporal-reasoning; do
    SRC="evaluation/results/lme_breadth_$QT"
    DST="evaluation/results/lme_qafix_$QT"
    rm -rf "$DST"; cp -r "$SRC" "$DST"
    QIDS=$(cd "$DST" && ls *.json | sed 's/\.json$//' | tr '\n' ' ')
    echo "[QAPATH] === requery $QT: $QIDS ==="
    python evaluation/LongMemEval/requery_cached.py --cache-dir "$DST" --qids $QIDS
    python evaluation/LongMemEval/score_longmemeval.py --cache-dir "$DST" \
      --output "evaluation/results/lme_qafix_${QT}_scores.json" --judge
  done
  echo "QAPATH_DONE $(date)"
} >> "$LOG" 2>&1
