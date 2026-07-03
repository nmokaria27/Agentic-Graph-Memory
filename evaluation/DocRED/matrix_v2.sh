#!/bin/bash
# Matrix v2 — after the stage-9 value-entity fix (MATRIX_REPORT.md rec #1).
# Re-runs only the affected strategies (rhf, hybrid — both use the organizer);
# singlepass caches stay valid. Rescoring covers all 6 strategy x slice combos.
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
A=evaluation/results/docred_kg_cache
B=evaluation/results/docred_kg_cache_heldout
OUT=evaluation/results
LOG=$OUT/matrix_v2.log

echo "[MATRIX2] start $(date)" >> "$LOG"

run() {  # run <slice-dir> <offset> <strategy>
    python -u evaluation/DocRED/run_eval.py --max-docs 5 --offset "$2" --strategy "$3" \
        --save-kg-dir "$1" --output "$OUT/docred_matrix2_$3_off$2.json" >> "$LOG" 2>&1
    echo "[MATRIX2] done: $3 offset=$2 $(date)" >> "$LOG"
}

run "$A" 0 hybrid
run "$A" 0 rhf
run "$B" 30 hybrid
run "$B" 30 rhf

for DIR in "$A" "$B"; do
    TAG=$( [ "$DIR" = "$A" ] && echo sliceA || echo sliceB )
    for S in rhf singlepass hybrid; do
        python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" \
            --output "$OUT/docred_scores_v2_${S}_${TAG}.json" >> "$LOG" 2>&1 \
        || python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" --no-embed \
            --output "$OUT/docred_scores_v2_${S}_${TAG}_noembed.json" >> "$LOG" 2>&1
    done
done

echo "[MATRIX2] MATRIX2_DONE $(date)" >> "$LOG"
