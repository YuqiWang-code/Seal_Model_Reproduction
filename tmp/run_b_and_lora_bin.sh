#!/bin/bash
# 1) B 基线（--no-m1，(RGB+T)/2，无任何可训练参数）
# 2) 已有 LoRA+LoRA3+BISF（旧全监督版预测）的二值化口径指标（Total 区第 22 行候选）
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

echo "=== B 基线推理（--no-m1） ==="
$PY -u test.py --no-m1 --checkpoint $R/results/Ours/BISF_60ep/best.pth \
  --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir $R/Seal_data/test --save_dir $R/results/Ours/B/test_maps 2>&1 | tail -3
echo "张数: $(ls $R/results/Ours/B/test_maps/*.png | wc -l)"

$PY - <<'PY'
from pathlib import Path
import cv2, numpy as np
for name in ("B",):
    src = Path(f"/share_datasets/Seal_Model_Reproduction/results/Ours/{name}/test_maps")
    dst = Path(f"/share_datasets/Seal_Model_Reproduction/results/Ours/{name}/test_maps_bin")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.png")):
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(dst / p.name), np.where(a >= 128, 255, 0).astype(np.uint8))
        n += 1
    print(name, "binarized", n)
PY

cd $R
echo "=== B 基线：原始概率 ==="
$PY evaluate.py --pred_root $R/results/Ours/B/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $R/results/Ours/B/metrics4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $R/results/Ours/B/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $R/results/Ours/B/metrics_small.json 2>&1 | grep -A5 '"metrics"'
echo "=== B 基线：二值化 ==="
$PY evaluate.py --pred_root $R/results/Ours/B/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $R/results/Ours/B/metrics4_bin.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $R/results/Ours/B/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $R/results/Ours/B/metrics_small_bin.json 2>&1 | grep -A5 '"metrics"'

echo "=== 已有 LoRA+LoRA3+BISF：二值化口径 ==="
FS=$R/results/Ours/FullySup
$PY - <<'PY'
from pathlib import Path
import cv2, numpy as np
src = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/FullySup/test_maps")
dst = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/FullySup/test_maps_bin")
dst.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(src.glob("*.png")):
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(dst / p.name), np.where(a >= 128, 255, 0).astype(np.uint8))
    n += 1
print("binarized", n)
PY
$PY evaluate.py --pred_root $FS/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $FS/metrics4_bin.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $FS/test_maps_bin --gt_dir $R/Seal_data/test/GT \
  --json_output $FS/metrics_small_bin.json 2>&1 | grep -A5 '"metrics"'
