#!/usr/bin/env python3
"""Object-level Precision-Recall curve (the textbook P-R shape) for Seal test.

For every threshold ``t`` the prediction is ``pred >= t``; 8-connected components are
matched to GT components exactly like ``metrics.AS1000Metrics`` /
``tools/target_curves.py`` (centroid distance strictly below 3 px, one-to-one,
raster order).  Then, accumulated over the whole test set:

    Recall(t)    = matched GT targets / all GT targets        (= Pd in the tables)
    Precision(t) = matched prediction components / all prediction components

``precision`` is the fraction of *objects* the method reports that are real targets,
so unlike the pixel-level P-R it is not dominated by the 0.02 % background: it starts
high at the highest threshold and falls when noise blobs appear.  The (0, 1) sentinel
and the "predict everything" endpoint are added by the plotter.

Usage:
    python tools/target_pr.py --maps results/DNANet/Seal/fullsup_test_maps \
        --gt .../Seal_data/test/GT --out results/target_pr/DNANet.json \
        --label DNA-Net --limit 400 --thresholds 255,248,240,232,224,216,208,200,192,184,176,168,160,144,128,112,96,64,32,1
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

STRUCT8 = ndimage.generate_binary_structure(2, 2)


def components(mask):
    labels, count = ndimage.label(mask, structure=STRUCT8)
    if count == 0:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    idx = np.arange(1, count + 1)
    centers = np.asarray(ndimage.center_of_mass(mask, labels, idx))
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return centers, areas


def parse_thresholds(spec):
    if spec == 'auto':
        return (list(range(255, 199, -4)) + list(range(200, 119, -8))
                + list(range(120, 39, -16)) + [32, 16, 8, 1])
    return sorted({int(v) for v in spec.split(',')}, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--thresholds', default='auto')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--match-distance', type=float, default=3.0)
    args = ap.parse_args()

    ths = parse_thresholds(args.thresholds)
    n_t = len(ths)
    names = sorted(n for n in os.listdir(args.gt) if n.lower().endswith('.png'))
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit(f'no GT PNGs in {args.gt}')

    n_gt = 0
    matched_gt = np.zeros(n_t, dtype=np.int64)
    n_pred_comp = np.zeros(n_t, dtype=np.int64)
    matched_comp = np.zeros(n_t, dtype=np.int64)

    for i, name in enumerate(names, 1):
        gt = np.asarray(Image.open(os.path.join(args.gt, name)).convert('L'))
        pred = np.asarray(Image.open(os.path.join(args.maps, name)).convert('L'))
        if gt.shape != pred.shape:
            raise SystemExit(f'size mismatch on {name}: {gt.shape} vs {pred.shape}')
        gt_centers, _ = components(gt > 127)
        n_gt += len(gt_centers)

        for k, t in enumerate(ths):
            centers, _ = components(pred >= t)
            n_pred_comp[k] += len(centers)
            matched = np.zeros(len(centers), dtype=bool)
            hit = 0
            for c in gt_centers:
                if len(centers) == 0:
                    break
                d = np.linalg.norm(centers - c, axis=1)
                cand = np.flatnonzero((d < args.match_distance) & ~matched)
                if cand.size:
                    matched[cand[0]] = True
                    hit += 1
            matched_gt[k] += hit
            matched_comp[k] += int(matched.sum())
        if i % 200 == 0:
            print(f'  {i}/{len(names)} {name}', flush=True)

    recall = (matched_gt / n_gt).tolist() if n_gt else [0.0] * n_t
    precision = [float(m) / n if n else 0.0
                 for m, n in zip(matched_comp.tolist(), n_pred_comp.tolist())]
    rec = {
        'label': args.label or os.path.basename(args.maps.rstrip('/')),
        'maps': args.maps, 'gt': args.gt,
        'images': len(names), 'n_gt_targets': int(n_gt),
        'thresholds': list(map(int, ths)),
        'recall': recall, 'precision': precision,
        'n_pred_components': n_pred_comp.tolist(),
        'matched_components': matched_comp.tolist(),
        'match_distance': args.match_distance, 'connectivity': 8,
        'level': 'object', 'gt_positive_rule': 'gt > 127',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)

    j = int(np.argmin(np.abs(np.array(ths) - 128)))
    print(f'{rec["label"]}: images={len(names)} GT targets={n_gt} '
          f'recall@t={ths[j]}={recall[j]:.5f} precision@t={ths[j]}={precision[j]:.5f}')
    for t, r, p in zip(ths, recall, precision):
        print(f'   t={t:3d} recall={r:7.4f} precision={p:7.4f}')
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
