#!/usr/bin/env python3
"""Target-level Pd / Fa curves (the "Pd vs Fa" ROC style used by IRSTD papers).

For every threshold t the prediction is ``pred >= t``; connected components are
matched to GT components exactly like ``metrics.AS1000Metrics`` (8-connectivity,
GT centroid distance strictly below 3 px, one-to-one, raster order), giving

    Pd(t) = matched GT targets / all GT targets
    Fa(t) = area of unmatched prediction components / all pixels

which are the same quantities as the Pd / Fa columns of the tables (at t = 128).

Usage:
    python tools/target_curves.py --maps results/DNANet/Seal/fullsup_test_maps \
        --gt /share_datasets/.../Seal_data/test/GT \
        --out results/target_curves/DNANet.json --label DNA-Net
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
    """Centres and areas of the 8-connected components, raster order."""
    labels, count = ndimage.label(mask, structure=STRUCT8)
    if count == 0:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    idx = np.arange(1, count + 1)
    centers = np.asarray(ndimage.center_of_mass(mask, labels, idx))
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return centers, areas


def default_thresholds():
    return (list(range(255, 199, -2)) + list(range(200, 119, -4))
            + list(range(120, 39, -8)) + list(range(40, 0, -8)) + [1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--thresholds', default='auto',
                    help='"auto" or comma separated list, e.g. 255,200,128,64,1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--match-distance', type=float, default=3.0)
    args = ap.parse_args()

    if args.thresholds == 'auto':
        ths = default_thresholds()
    else:
        ths = sorted({int(v) for v in args.thresholds.split(',')}, reverse=True)
    n_t = len(ths)

    names = sorted(n for n in os.listdir(args.gt) if n.lower().endswith('.png'))
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit(f'no GT PNGs in {args.gt}')

    n_gt = 0
    detected = np.zeros(n_t, dtype=np.int64)
    fa_pixels = np.zeros(n_t, dtype=np.int64)
    total_pixels = 0

    for i, name in enumerate(names, 1):
        gt = np.asarray(Image.open(os.path.join(args.gt, name)).convert('L'))
        pred = np.asarray(Image.open(os.path.join(args.maps, name)).convert('L'))
        if gt.shape != pred.shape:
            raise SystemExit(f'size mismatch on {name}: {gt.shape} vs {pred.shape}')
        gt_mask = gt > 127
        gt_centers, _ = components(gt_mask)
        n_gt += len(gt_centers)
        total_pixels += gt_mask.size

        for k, t in enumerate(ths):
            pred_mask = pred >= t
            centers, areas = components(pred_mask)
            matched = np.zeros(len(centers), dtype=bool)
            for c in gt_centers:
                if len(centers) == 0:
                    break
                d = np.linalg.norm(centers - c, axis=1)
                cand = np.flatnonzero((d < args.match_distance) & ~matched)
                if cand.size:
                    matched[cand[0]] = True
            detected[k] += int(matched.sum())
            fa_pixels[k] += int(areas[~matched].sum())
        if i % 200 == 0:
            print(f'  {i}/{len(names)} {name}', flush=True)

    pd = (detected / n_gt).tolist() if n_gt else [0.0] * n_t
    fa = (fa_pixels / total_pixels).tolist() if total_pixels else [0.0] * n_t
    rec = {
        'label': args.label or os.path.basename(args.maps.rstrip('/')),
        'maps': args.maps, 'gt': args.gt,
        'images': len(names), 'n_gt_targets': int(n_gt),
        'thresholds': list(map(int, ths)),
        'Pd': pd, 'Fa': fa,
        'match_distance': args.match_distance, 'connectivity': 8,
        'gt_positive_rule': 'gt > 127',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)

    j = ths.index(128) if 128 in ths else int(np.argmin(np.abs(np.array(ths) - 128)))
    print(f'{rec["label"]}: images={len(names)} GT targets={n_gt} '
          f'Pd@t={ths[j]}={pd[j]:.5f} Fa@t={ths[j]}={fa[j]:.3e}')
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
