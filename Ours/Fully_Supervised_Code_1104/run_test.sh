#!/usr/bin/env bash
# B+BISF：导出 Seal test 预测并按项目统一口径算八项指标。
set -euo pipefail
cd "$(dirname "$0")"

ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
export SEAL_CUDA=${SEAL_CUDA:-1}
OUT=$ROOT/results/Ours/BISF

$PY test.py \
  --checkpoint $OUT/best.pth \
  --sam_checkpoint $ROOT/Ours/weights/sam3.pt \
  --test_data_dir $ROOT/Seal_data/test \
  --save_dir $OUT/test_maps

echo "预测张数: $(ls $OUT/test_maps/*.png | wc -l)"

cd $ROOT
$PY evaluate.py --pred_root $OUT/test_maps --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $OUT/metrics4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $OUT/test_maps --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $OUT/metrics_small.json 2>&1 | grep -A6 '"metrics"'
