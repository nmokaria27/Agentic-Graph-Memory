#!/usr/bin/env bash
# After run_governed_vs_ungoverned.py finishes (v2 10-doc), split both KGs
# per-document and run F1 eval against the SciERC gold. Writes:
#   evaluation/results/f1_governed_10docs_v2.json
#   evaluation/results/f1_ungoverned_10docs_v2.json
set -euo pipefail
cd "$(dirname "$0")/.."

for side in governed ungoverned; do
  kg_in="evaluation/results/${side}_created_10docs_v2.json"
  perdoc="evaluation/results/${side}_10docs_v2_perdoc"
  mkdir -p "$perdoc"
  python3 scripts/split_kg_by_doc.py --input "$kg_in" --output-dir "$perdoc"
  python3 evaluation/run_evaluation.py \
    --skip-pipeline --results-dir "$perdoc" \
    --max-docs 10 --fixed-schema \
    --output-json "evaluation/results/f1_${side}_10docs_v2.json"
done

echo "=== governed v2 ==="
python3 -c "import json; d=json.load(open('evaluation/results/f1_governed_10docs_v2.json')); print('entity_strict F1:', d['entity_strict']['f1'], '  triple_strict F1:', d['triple_strict']['f1'])"
echo "=== ungoverned v2 ==="
python3 -c "import json; d=json.load(open('evaluation/results/f1_ungoverned_10docs_v2.json')); print('entity_strict F1:', d['entity_strict']['f1'], '  triple_strict F1:', d['triple_strict']['f1'])"
