#!/bin/bash
# 逐图 min-max 版本对照评估（排查 Fw 与表里 B 行的口径差异）
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/BISF

$PY /tmp/make_minmax_maps.py
echo "=== min-max 版本 四项 ==="
cd $R
$PY evaluate.py --pred_root $OUT/test_maps_minmax --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4_minmax.json 2>&1 | tail -8
echo "=== min-max 版本 小目标四项 ==="
$PY evaluate_small_targets.py --pred_root $OUT/test_maps_minmax --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small_minmax.json 2>&1 | grep -A6 '"metrics"'
echo "=== 原始图统计（前 20 张） ==="
$PY - <<'PY'
from pathlib import Path
import cv2, numpy as np
src = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps")
mx, mn, mean = [], [], []
for p in sorted(src.glob("*.png"))[:20]:
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    mx.append(int(a.max())); mn.append(int(a.min())); mean.append(float(a.mean()))
print("max:", mx[:10])
print("mean:", [round(m, 2) for m in mean[:10]])
PY
