#!/bin/bash
# Matrix v3 — hybrid v2 changes (HYBRID_V2_RUN.md): stage-9 entity funnel fixes
# (year-safe id cleanup, alias merge on collision, per-reason drop logging) +
# wide-harvest relation seeding. Re-runs rhf + hybrid (both traverse changed
# code); singlepass caches stay valid as the untouched control.
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
A=evaluation/results/docred_kg_cache
B=evaluation/results/docred_kg_cache_heldout
OUT=evaluation/results
LOG=$OUT/matrix_v3.log

echo "[MATRIX3] start $(date)" >> "$LOG"

# Pipeline changed — stale rhf/hybrid caches would silently reuse v2 outputs.
rm -f "$A"/doc_*_rhf.json "$A"/doc_*_hybrid.json \
      "$B"/doc_*_rhf.json "$B"/doc_*_hybrid.json
echo "[MATRIX3] stale rhf/hybrid caches cleared" >> "$LOG"

run() {  # run <slice-dir> <offset> <strategy>
    python -u evaluation/DocRED/run_eval.py --max-docs 5 --offset "$2" --strategy "$3" \
        --save-kg-dir "$1" --output "$OUT/docred_matrix3_$3_off$2.json" >> "$LOG" 2>&1
    echo "[MATRIX3] MILESTONE done: $3 offset=$2 $(date)" >> "$LOG"
}

run "$A" 0 hybrid
run "$A" 0 rhf
run "$B" 30 hybrid
run "$B" 30 rhf

for DIR in "$A" "$B"; do
    TAG=$( [ "$DIR" = "$A" ] && echo sliceA || echo sliceB )
    for S in rhf singlepass hybrid; do
        python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" \
            --output "$OUT/docred_scores_v3_${S}_${TAG}.json" >> "$LOG" 2>&1 \
        || python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" --no-embed \
            --output "$OUT/docred_scores_v3_${S}_${TAG}_noembed.json" >> "$LOG" 2>&1
    done
done

echo "[MATRIX3] MATRIX3_DONE $(date)" >> "$LOG"
