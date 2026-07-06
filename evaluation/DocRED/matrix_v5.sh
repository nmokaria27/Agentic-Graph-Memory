#!/bin/bash
# Matrix v4 — labels measurement fix (run_eval dumps entity.labels; scorer matches
# name UNION labels). Coref renames entities to canonical ids; the gold-matchable
# surfaces live in labels, previously dropped from the dump. This fix is a SCORING
# change, orthogonal to extraction — run alone so any hybrid recall gain is cleanly
# attributable to it (not confounded with a coref change).
# Re-extract hybrid + rhf (need labels in dump); singlepass cache reused (scorer
# falls back to name for label-less caches, so its scores are unchanged).
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
A=evaluation/results/docred_kg_cache
B=evaluation/results/docred_kg_cache_heldout
OUT=evaluation/results
LOG=$OUT/matrix_v5.log

echo "[MATRIX5] start $(date)" >> "$LOG"
rm -f "$A"/doc_*_rhf.json "$A"/doc_*_hybrid.json \
      "$B"/doc_*_rhf.json "$B"/doc_*_hybrid.json
echo "[MATRIX5] stale rhf/hybrid caches cleared (singlepass kept)" >> "$LOG"

run() {  # run <slice-dir> <offset> <strategy>
    python -u evaluation/DocRED/run_eval.py --max-docs 5 --offset "$2" --strategy "$3" \
        --save-kg-dir "$1" --output "$OUT/docred_matrix5_$3_off$2.json" >> "$LOG" 2>&1
    echo "[MATRIX5] MILESTONE done: $3 offset=$2 $(date)" >> "$LOG"
}

run "$A" 0 hybrid
run "$A" 0 rhf
run "$B" 30 hybrid
run "$B" 30 rhf

for DIR in "$A" "$B"; do
    TAG=$( [ "$DIR" = "$A" ] && echo sliceA || echo sliceB )
    for S in rhf singlepass hybrid; do
        python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" \
            --output "$OUT/docred_scores_v5_${S}_${TAG}.json" >> "$LOG" 2>&1 \
        || python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" --no-embed \
            --output "$OUT/docred_scores_v5_${S}_${TAG}_noembed.json" >> "$LOG" 2>&1
    done
done

echo "[MATRIX5] MATRIX5_DONE $(date)" >> "$LOG"
