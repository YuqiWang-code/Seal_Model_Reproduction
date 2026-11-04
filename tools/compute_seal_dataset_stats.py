"""Seal 数据集统计：每图目标个数、每个目标（8 连通域）的像素面积。

在服务器上运行（只读 GT，不改数据）：
    python seal_dataset_stats.py > seal_dataset_stats.json
"""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/share_datasets/Seal_Model_Reproduction/Seal_data")
SPLITS = ["train", "val", "test"]


def hist_summary(area_hist: Counter):
    if not area_hist:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p99": 0.0, "max": 0}
    values, counts = zip(*sorted(area_hist.items()))
    arr = np.repeat(np.asarray(values), np.asarray(counts))
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def stats_for(gt_dir: Path):
    per_image = Counter()
    area_hist = Counter()
    empty = 0
    for path in sorted(gt_dir.glob("*.png")):
        gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"cannot read {path}")
        mask = (gt > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = n - 1
        per_image[count] += 1
        if count == 0:
            empty += 1
        for area in stats[1:, cv2.CC_STAT_AREA]:
            area_hist[int(area)] += 1
    return {
        "images": sum(per_image.values()),
        "targets": sum(area_hist.values()),
        "empty_images": empty,
        "per_image": {int(k): int(v) for k, v in sorted(per_image.items())},
        "area_hist": {int(k): int(v) for k, v in sorted(area_hist.items())},
        "area_summary": hist_summary(area_hist),
    }


def main():
    out = {}
    total_images = Counter()
    total_areas = Counter()
    for split in SPLITS:
        gt_dir = ROOT / split / "GT"
        if not gt_dir.is_dir():
            raise SystemExit(f"missing {gt_dir}")
        out[split] = stats_for(gt_dir)
        total_images.update(out[split]["per_image"])
        total_areas.update(out[split]["area_hist"])
    out["all"] = {
        "images": sum(total_images.values()),
        "targets": sum(total_areas.values()),
        "empty_images": sum(v for k, v in total_images.items() if k == 0),
        "per_image": {int(k): int(v) for k, v in sorted(total_images.items())},
        "area_hist": {int(k): int(v) for k, v in sorted(total_areas.items())},
        "area_summary": hist_summary(total_areas),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
