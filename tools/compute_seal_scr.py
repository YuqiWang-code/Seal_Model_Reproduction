"""Seal 数据集 SCR（信杂比）统计。

SCR = |mean(target) - mean(background)| / std(background)，在**原始 16 位热图**上
计算（逐图 min-max 到 8 位是线性变换，SCR 不变；用 16 位可避免量化误差）。

背景取目标包围框向外扩一倍（窗口边长 = 3 × 包围框边长）的局部区域，去掉目标像素；
局部背景不足 20 像素或标准差为 0 时退回全图背景（除目标外）。

输出：
    per_image : 每张图的 SCR（多目标时取该图的最大 SCR），用于"横轴 SCR、纵轴图片数"
    per_target: 每个目标的 SCR（附带统计量，便于换成 Fig. 4(c) 那种按目标统计的版本）

服务器上运行（只读 Seal_data）：
    python compute_seal_scr.py > seal_scr_stats.json
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/share_datasets/Seal_Model_Reproduction/Seal_data")
SPLITS = ["train", "val", "test"]


def image_scr(img, mask):
    """Return (per-target SCR list, bounding-box areas) for one image."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    scores = []
    areas = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        comp = labels == k
        mu_t = float(img[comp].mean())
        pad = max(w, h)
        y0, y1 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(img.shape[1], x + w + pad)
        win = img[y0:y1, x0:x1].astype(np.float64)
        winmask = comp[y0:y1, x0:x1]
        bg = win[~winmask]
        if bg.size < 20 or bg.std() < 1e-6:
            bg = img[~mask].astype(np.float64)
        sd = float(bg.std())
        if sd < 1e-6:
            scores.append(0.0)
        else:
            scores.append(abs(mu_t - float(bg.mean())) / sd)
        areas.append(int(area))
    return scores, areas


def main():
    out = {}
    all_image = []
    all_target = []
    for split in SPLITS:
        t_dir = ROOT / split / "T"
        g_dir = ROOT / split / "GT"
        per_image = []
        per_target = []
        for gt_path in sorted(g_dir.glob("*.png")):
            tif = t_dir / f"{gt_path.stem}.tif"
            img = cv2.imread(str(tif), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"cannot read {tif}")
            if img.ndim == 3:
                img = img[..., 0]
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            mask = gt > 127
            scores, _ = image_scr(img.astype(np.float64), mask)
            if scores:
                per_image.append(float(max(scores)))
                per_target.extend(float(s) for s in scores)
        arr_i = np.asarray(per_image)
        arr_t = np.asarray(per_target)
        out[split] = {
            "images": len(per_image),
            "targets": len(per_target),
            "per_image": [round(float(v), 4) for v in per_image],
            "summary_image": {
                "min": float(arr_i.min()), "p25": float(np.percentile(arr_i, 25)),
                "median": float(np.median(arr_i)), "mean": float(arr_i.mean()),
                "p75": float(np.percentile(arr_i, 75)), "p95": float(np.percentile(arr_i, 95)),
                "max": float(arr_i.max()),
            },
            "summary_target": {
                "min": float(arr_t.min()), "median": float(np.median(arr_t)),
                "mean": float(arr_t.mean()), "p95": float(np.percentile(arr_t, 95)),
                "max": float(arr_t.max()),
            },
        }
        all_image.extend(per_image)
        all_target.extend(per_target)
    arr_i = np.asarray(all_image)
    arr_t = np.asarray(all_target)
    out["all"] = {
        "images": len(all_image),
        "targets": len(all_target),
        "per_image": [round(float(v), 4) for v in all_image],
        "summary_image": {
            "min": float(arr_i.min()), "p25": float(np.percentile(arr_i, 25)),
            "median": float(np.median(arr_i)), "mean": float(arr_i.mean()),
            "p75": float(np.percentile(arr_i, 75)), "p95": float(np.percentile(arr_i, 95)),
            "max": float(arr_i.max()),
        },
        "summary_target": {
            "min": float(arr_t.min()), "median": float(np.median(arr_t)),
            "mean": float(arr_t.mean()), "p95": float(np.percentile(arr_t, 95)),
            "max": float(arr_t.max()),
        },
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
