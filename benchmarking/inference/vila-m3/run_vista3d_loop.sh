#!/usr/bin/env bash
set -euo pipefail

DATA=${1:-"3D_VLM_Spatial/qa_generation_v2/spatial_qa_filtered_full_nifti.jsonl"}
OUT=${2:-"3D_VLM_Spatial/reports/vila_m3_vista3d_predictions_full.jsonl"}
ROOT=${3:-"3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed"}
MODEL=${4:-"MONAI/Llama3-VILA-M3-8B"}
STAGE=${5:-expert-only}
CACHE_DIR=${6:-"3D_VLM_Spatial/reports/vista3d_cache"}

TOTAL=$(wc -l < "$DATA")

echo "Dataset: $DATA"
echo "Output:  $OUT"
echo "Root:    $ROOT"
echo "Model:   $MODEL"
if [[ "$CHUNK" -gt 0 ]]; then
  echo "Chunk:   $CHUNK"
else
  echo "Chunk:   full (no limit)"
fi
echo "Stage:   $STAGE"
echo "Cache:   $CACHE_DIR"
echo "Total:   $TOTAL"

while true; do
  DONE=$(find "$CACHE_DIR" -name "vista3d_summary.json" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$DONE" -ge "$TOTAL" ]]; then
    echo "All done (cached): $DONE / $TOTAL"
    break
  fi

  echo "Cached: $DONE / $TOTAL (running expert-only)"
  CMD=(python benchmarking/inference/vila-m3/run_vista3d_eval.py
    --dataset "$DATA"
    --nifti-root "$ROOT"
    --output "$OUT"
    --model-path "$MODEL"
    --stage "$STAGE"
  )
  "${CMD[@]}" || echo "crashed, retrying..."
done
