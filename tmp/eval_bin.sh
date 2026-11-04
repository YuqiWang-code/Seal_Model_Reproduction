#!/bin/bash
# 用二值化（>=128）预测图评估，排查表里 B 行是否用了二值图口径
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/BISF
SRC=$OUT/test_maps
DST=$OUT/test_maps_bin

$PY - <<'PY'
from pathlib import Path
import cv2, numpy as np
src = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps")
dst = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps_bin")
dst.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(src.glob("*.png")):
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(dst / p.name), np.where(a >= 128, 255, 0).astype(np.uint8))
    n += 1
print("binarized", n)
PY

cd $R
$PY evaluate.py --pred_root $DST --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4_bin.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $DST --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small_bin.json 2>&1 | grep -A6 '"metrics"'
