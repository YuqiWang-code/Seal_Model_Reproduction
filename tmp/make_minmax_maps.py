"""把 B+BISF 的原始概率图做逐图 min-max 归一化，看 Fw/MAE 是否与表里 B 行的口径一致。

原始 test.py 直接保存 sigmoid 概率（可能是暗图）；Original_Code 的 test.py 则做逐图 min-max。
这里生成一份 min-max 版本用于对照评估（不改动原始预测）。
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

src = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps")
dst = Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps_minmax")
dst.mkdir(parents=True, exist_ok=True)

count = 0
for p in sorted(src.glob("*.png")):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        out = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
    else:
        out = np.zeros_like(img, dtype=np.float32)
    cv2.imwrite(str(dst / p.name), np.rint(out).astype(np.uint8))
    count += 1
print("wrote", count, "min-max maps to", dst)
