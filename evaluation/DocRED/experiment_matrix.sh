#!/bin/bash
# Phase 3 experiment matrix (post coref-fix): all strategy x slice combinations.
#   Slice A (docs 0-4):  diagnostic slice every fix was developed against.
#   Slice B (docs 30-34): HELD-OUT slice — never hand-read, never tuned on.
#                         Guards against fixes that memorize the diagnostic docs.
# Strategies: rhf (fixed coref), hybrid (wide harvest -> deliberation), singlepass.
# Singlepass docs 0-4 already cached; per-doc checkpoints make every step resumable.
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
A=evaluation/results/docred_kg_cache
B=evaluation/results/docred_kg_cache_heldout
OUT=evaluation/results
LOG=$OUT/matrix_run.log
mkdir -p "$B"

echo "[MATRIX] start $(date)" >> "$LOG"

# don't overlap with the in-flight doc-3 validation or hybrid smoke
while pgrep -f "repro_coref.py|--max-docs 1 --strategy hybrid" > /dev/null; do sleep 30; done
echo "[MATRIX] predecessors clear $(date)" >> "$LOG"

run() {  # run <slice-dir> <offset> <strategy>
    python -u evaluation/DocRED/run_eval.py --max-docs 5 --offset "$2" --strategy "$3" \
        --save-kg-dir "$1" --output "$OUT/docred_matrix_$3_off$2.json" >> "$LOG" 2>&1
    echo "[MATRIX] done: $3 offset=$2 $(date)" >> "$LOG"
}

# Slice A — cheapest first so partial results are useful early
run "$A" 0 hybrid
run "$A" 0 rhf
# Slice B — held-out: all three strategies
run "$B" 30 singlepass
run "$B" 30 hybrid
run "$B" 30 rhf

# Scoring: every strategy on both slices (embeddings; token-overlap fallback)
for DIR in "$A" "$B"; do
    TAG=$( [ "$DIR" = "$A" ] && echo sliceA || echo sliceB )
    for S in rhf singlepass hybrid; do
        python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" \
            --output "$OUT/docred_scores_${S}_${TAG}.json" >> "$LOG" 2>&1 \
        || python evaluation/DocRED/score_docred.py --kg-dir "$DIR" --strategy "$S" --no-embed \
            --output "$OUT/docred_scores_${S}_${TAG}_noembed.json" >> "$LOG" 2>&1
    done
done

echo "[MATRIX] MATRIX_DONE $(date)" >> "$LOG"
