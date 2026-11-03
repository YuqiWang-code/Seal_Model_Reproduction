#!/usr/bin/env bash
set -euo pipefail

ROOT=/share_datasets/Seal_Model_Reproduction
METHOD=$ROOT/NESS-Net-main
OUT=$ROOT/results/NESS-Net/Seal
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

cd "$METHOD"
for STAGE in general refinement; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u infer_seal.py \
    --seal-dir "$ROOT/Seal_data" \
    --checkpoint "$METHOD/NESS_${STAGE}.pth" \
    --stage "$STAGE" \
    --rgb-edge-dir "$OUT/rgb_namlab_edges" \
    --thermal-edge-dir "$OUT/thermal_namlab_edges" \
    --output-dir "$OUT/${STAGE}_namlab_test_maps" \
    > "$OUT/${STAGE}_namlab_infer.log" 2>&1

  "$PY" "$ROOT/evaluate.py" \
    --pred_root "$OUT/${STAGE}_namlab_test_maps" \
    --gt_dir "$ROOT/Seal_data/test/GT" \
    > "$OUT/${STAGE}_namlab_evaluate.log" 2>&1
done
