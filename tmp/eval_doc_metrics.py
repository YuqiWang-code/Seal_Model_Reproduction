"""Reproduce the AS1000Metrics variant from docs/小目标评价指标.pdf (the draft).

Differences from the project's metrics.py, kept exactly as the draft does them:
  * matching: all pairs with distance <= match_distance, globally sorted by
    distance, greedy first-come (project uses "first unmatched prediction in
    GT raster order" and a strict "<");
  * Fa numerator: pixel-level false positives (project uses the area of
    unmatched predicted components);
  * output: percentages (x100).

One deliberate deviation: the draft binarizes with `pred > 0` because it was
written for 0/255 masks. These prediction PNGs are continuous, so a plain
`> 0` would mark essentially every pixel; the same 0.5 threshold as the
project is used, otherwise IoU would be meaningless (~2e-4).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

pred_dir, gt_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MATCH_DISTANCE = 3
THRESHOLD = 0.5

tp = fp = fn = 0
n_gt = n_detected = 0
plus_3 = 0          # detections if the draft's "<=" were used instead of "<"
false_pixels = 0
n_pixels = 0


def components(mask):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(centroids[i], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]


for index, gt_path in enumerate(sorted(gt_dir.glob('*.png')), 1):
    pred = cv2.imread(str(pred_dir / gt_path.name), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise SystemExit(f'cannot read pair {gt_path.name}')
    if pred.shape != gt.shape:
        raise SystemExit(f'shape mismatch {gt_path.name}: {pred.shape} vs {gt.shape}')

    prob = pred.astype(np.float64) / 255.0
    p = prob > THRESHOLD
    g = gt > 128

    tp += int((p & g).sum())
    fp += int((p & ~g).sum())
    fn += int((~p & g).sum())
    false_pixels += int((p & ~g).sum())
    n_pixels += int(g.size)

    pred_objects = components(p)
    gt_objects = components(g)
    n_gt += len(gt_objects)

    candidates = []
    for gi, (gc, _) in enumerate(gt_objects):
        for pi, (pc, _) in enumerate(pred_objects):
            d = float(np.linalg.norm(np.asarray(gc) - np.asarray(pc)))
            if d <= MATCH_DISTANCE:
                candidates.append((d, gi, pi))
    candidates.sort(key=lambda x: x[0])
    matched_gt, matched_pred = set(), set()
    for _, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
    n_detected += len(matched_gt)
    if index % 500 == 0:
        print(f'{index}', flush=True)

eps = 1e-10
iou = tp / (tp + fp + fn + eps)
f1 = 2 * tp / (2 * tp + fp + fn + eps)
pd = n_detected / (n_gt + eps)
fa = false_pixels / (n_pixels + eps)

record = {
    'variant': 'docs/小目标评价指标.pdf draft algorithm (percent output)',
    'notes': 'Fa = pixel-level FP / total pixels; matching distance <=3, distance-sorted greedy; binarization 0.5 (draft used pred>0 on 0/255 masks)',
    'counts': {'n_images': 1983, 'tp': tp, 'fp': fp, 'fn': fn, 'n_gt': n_gt,
               'n_detected': n_detected, 'false_pixels': false_pixels, 'n_pixels': n_pixels},
    'percent': {'IoU': iou * 100, 'F1': f1 * 100, 'Pd': pd * 100, 'Fa': fa * 100},
    'fraction': {'IoU': iou, 'F1': f1, 'Pd': pd, 'Fa': fa},
}
out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record['percent'], indent=2))
