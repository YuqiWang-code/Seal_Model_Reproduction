#!/bin/bash
# B+BISF 60 epoch：推理 + 两种口径（原始概率 / 二值化）八项指标
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/BISF_60ep
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

echo "=== 训练结论 ==="; grep "IoU:" $R/logs/ours_bisf_60ep.log | tail -2
ls -la $OUT/best.pth

echo "=== 推理 Sealtest 1983 ==="
$PY -u test.py --checkpoint $OUT/best.pth \
  --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir $R/Seal_data/test --save_dir $OUT/test_maps 2>&1 | tail -2
echo "预测张数: $(ls $OUT/test_maps/*.png | wc -l)"

echo "=== 二值化副本 ==="
$PY - <<'PY'
from pathlib import Path
import cv2, numpy as np
src = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF_60ep/test_maps")
dst = src.parent / "test_maps_bin"
dst.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(src.glob("*.png")):
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(dst / p.name), np.where(a >= 128, 255, 0).astype(np.uint8))
    n += 1
print("binarized", n)
PY

cd $R
echo "=== 原始概率：四项 ==="
$PY evaluate.py --pred_root $OUT/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4.json 2>&1 | tail -8
echo "=== 原始概率：小目标四项 ==="
$PY evaluate_small_targets.py --pred_root $OUT/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small.json 2>&1 | grep -A6 '"metrics"'
echo "=== 二值化：四项 ==="
$PY evaluate.py --pred_root $OUT/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4_bin.json 2>&1 | tail -8
echo "=== 二值化：小目标四项 ==="
$PY evaluate_small_targets.py --pred_root $OUT/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small_bin.json 2>&1 | grep -A6 '"metrics"'
