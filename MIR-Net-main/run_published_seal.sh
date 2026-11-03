#!/usr/bin/env bash
set -euo pipefail

ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
METHOD=$ROOT/MIR-Net-main
OUT=$ROOT/results/MIR-Net/Seal/published_test_maps

cd "$METHOD"
CUDA_VISIBLE_DEVICES=0 "$PY" -u infer_seal.py \
  --data "$ROOT/Seal_data/test" \
  --checkpoint "$METHOD/published_checkpoint/31_wM_test1/model-best.pth" \
  --output "$OUT"

"$PY" "$ROOT/evaluate.py" \
  --pred_root "$OUT" \
  --gt_dir "$ROOT/Seal_data/test/GT"
