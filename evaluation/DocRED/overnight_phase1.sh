#!/bin/bash
# Overnight DocRED Phases 0-2 (see PLAN.md): fully self-contained, no supervision.
# - waits for any in-flight Phase 0 smoke to finish (cache-skip makes rerun safe)
# - RHF docs 0-4, then singlepass docs 0-4 (per-doc checkpoints -> crash loses <=1 doc)
# - scores both strategies offline (embedding mode; token-overlap fallback if gpu01 is down)
set -u
cd /home/nmokaria/Agentic-Graph-Memory
set -a; . ./.env 2>/dev/null; set +a
export NEMOTRON_THINKING=on
KG=evaluation/results/docred_kg_cache
OUT=evaluation/results
LOG=$OUT/overnight_phase1.log

echo "[OVERNIGHT] start $(date)" >> "$LOG"

# 1. let the in-flight Phase 0 smoke finish (it writes doc_0_rhf.json)
while pgrep -f "run_eval.py --max-docs 1" > /dev/null; do sleep 30; done
echo "[OVERNIGHT] phase0 clear $(date)" >> "$LOG"

# 2. Phase 1: RHF diagnostic slice (doc 0 skipped if cached)
python -u evaluation/DocRED/run_eval.py --max-docs 5 --strategy rhf \
    --save-kg-dir "$KG" --output "$OUT/docred_phase1_rhf.json" >> "$LOG" 2>&1
echo "[OVERNIGHT] rhf slice done $(date)" >> "$LOG"

# 3. Phase 2 counterpart: singlepass on the SAME slice
python -u evaluation/DocRED/run_eval.py --max-docs 5 --strategy singlepass \
    --save-kg-dir "$KG" --output "$OUT/docred_phase1_singlepass.json" >> "$LOG" 2>&1
echo "[OVERNIGHT] singlepass slice done $(date)" >> "$LOG"

# 4. offline scoring (embeddings; fall back to token overlap if embed server is down)
for S in rhf singlepass; do
    python evaluation/DocRED/score_docred.py --kg-dir "$KG" --strategy "$S" \
        --output "$OUT/docred_scores_$S.json" >> "$LOG" 2>&1 \
    || python evaluation/DocRED/score_docred.py --kg-dir "$KG" --strategy "$S" --no-embed \
        --output "$OUT/docred_scores_${S}_noembed.json" >> "$LOG" 2>&1
done

echo "[OVERNIGHT] OVERNIGHT_DONE $(date)" >> "$LOG"
