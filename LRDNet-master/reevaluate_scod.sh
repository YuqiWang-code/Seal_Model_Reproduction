#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=/home/yqwang/miniforge3/envs/lrdnet_seal/bin/python
PROJECT=/share_datasets/Seal_Model_Reproduction
EXPERIMENT=experiments/SCOD

"$PYTHON" "$PROJECT/evaluate.py" \
  --pred_root "$PWD/$EXPERIMENT/test_maps" \
  --gt_dir "$PROJECT/Seal_data/test/GT" \
  --json_output "$EXPERIMENT/Seal_original_metrics.json" \
  > "$EXPERIMENT/Seal_original_evaluate.log" 2>&1

for SOURCE_TEST in CAMO COD10K; do
  "$PYTHON" "$PROJECT/evaluate.py" \
    --pred_root "$PWD/$EXPERIMENT/$SOURCE_TEST/test_maps" \
    --gt_dir "$PWD/data_original/TestDataset/$SOURCE_TEST/GT" \
    --json_output "$EXPERIMENT/${SOURCE_TEST}_original_metrics.json" \
    > "$EXPERIMENT/${SOURCE_TEST}_original_evaluate.log" 2>&1
done
