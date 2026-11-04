#!/bin/bash
# 等原始版试跑推理完成，再算八项指标（仅预览，最终会用训练结束后的 best3.pth 重跑）
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
MAPS=$ROOT/results/Ours/Original/infer_dryrun/best3/test
OUT=$ROOT/results/Ours/Original/infer_dryrun

for i in $(seq 1 40); do
  n=$(ls "$MAPS"/*.png 2>/dev/null | wc -l)
  if [ "$n" -ge 1983 ]; then break; fi
  sleep 30
done
echo "predictions: $(ls "$MAPS"/*.png | wc -l)"

cd "$ROOT"
$PY evaluate.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics4.json" 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics_small.json" 2>&1 | grep -A6 '"metrics"'
