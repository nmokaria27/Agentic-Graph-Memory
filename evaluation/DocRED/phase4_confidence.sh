#!/bin/bash
# Phase 4 confidence run (ROADMAP Phase A #3 / MATRIX_REPORT v5 decision):
# 40 fresh, never-touched dev docs (100-139), hybrid v2 (frozen extractor) +
# singlepass (control). Purpose: confirm the slice-A picture at a sample size
# where run-to-run variance (which swamped n=5 deltas) cannot dominate, and
# hand the memory benchmarks a stable extraction baseline.
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
C=evaluation/results/docred_kg_cache_phase4
OUT=evaluation/results
LOG=$OUT/phase4_confidence.log

echo "[PHASE4] start $(date)" >> "$LOG"

run() {  # run <strategy>
    python -u evaluation/DocRED/run_eval.py --max-docs 40 --offset 100 --strategy "$1" \
        --save-kg-dir "$C" --output "$OUT/docred_phase4_$1.json" >> "$LOG" 2>&1
    echo "[PHASE4] MILESTONE done: $1 $(date)" >> "$LOG"
}

run singlepass   # fast control first (~2h) so early numbers exist
run hybrid       # frozen extractor (~12h)

for S in singlepass hybrid; do
    python evaluation/DocRED/score_docred.py --kg-dir "$C" --strategy "$S" \
        --output "$OUT/docred_scores_phase4_${S}.json" >> "$LOG" 2>&1 \
    || python evaluation/DocRED/score_docred.py --kg-dir "$C" --strategy "$S" --no-embed \
        --output "$OUT/docred_scores_phase4_${S}_noembed.json" >> "$LOG" 2>&1
done

echo "[PHASE4] PHASE4_DONE $(date)" >> "$LOG"
