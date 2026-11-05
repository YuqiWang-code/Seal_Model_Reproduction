#!/usr/bin/env python3
"""Accumulate prediction-value histograms for ROC / PR curves, one method at a time.

For every test image we split the pixels into GT foreground (``gt > 127``, the
convention used by ``evaluate_small_targets.py``) and GT background, then bin the
8-bit prediction value (0-255) of those pixels into two 256-bin histograms. From
the two accumulated histograms any pixel-level threshold gives

    TPR(t) = sum(pos_hist[t:]) / n_pos
    FPR(t) = sum(neg_hist[t:]) / n_neg
    Precision(t) = TP(t) / (TP(t) + FP(t))

so a single small JSON per method is enough to redraw every ROC/PR curve without
shipping the 1,983 prediction PNGs around.

Usage:
    python tools/roc_hist.py --maps results/DNANet/Seal/fullsup_test_maps \
        --gt /share_datasets/Seal_Model_Reproduction/Seal_data/test/GT \
        --out results/roc_hist/DNANet.json --label DNA-Net
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

BINS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True, help='directory of prediction PNGs (8-bit gray)')
    ap.add_argument('--gt', required=True, help='directory of GT PNGs, same file names')
    ap.add_argument('--out', required=True, help='output JSON path')
    ap.add_argument('--label', default=None, help='display label stored in the JSON')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N images')
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.gt) if n.lower().endswith('.png'))
    if args.limit:
        names = names[:args.limit]
    if not names:
        sys.exit(f'no GT PNGs in {args.gt}')

    pos_hist = np.zeros(BINS, dtype=np.int64)
    neg_hist = np.zeros(BINS, dtype=np.int64)
    missing = []
    shapes = set()

    for i, name in enumerate(names, 1):
        map_path = os.path.join(args.maps, name)
        if not os.path.exists(map_path):
            missing.append(name)
            continue
        gt = np.asarray(Image.open(os.path.join(args.gt, name)).convert('L'))
        pred = np.asarray(Image.open(map_path).convert('L'))
        if gt.shape != pred.shape:
            raise SystemExit(f'size mismatch on {name}: gt {gt.shape} vs pred {pred.shape}')
        shapes.add(gt.shape)
        fg = gt > 127
        bg = ~fg
        pos_hist += np.bincount(pred[fg], minlength=BINS)[:BINS]
        neg_hist += np.bincount(pred[bg], minlength=BINS)[:BINS]
        if i % 400 == 0:
            print(f'  {i}/{len(names)} {name}', flush=True)

    n_pos = int(pos_hist.sum())
    n_neg = int(neg_hist.sum())
    n_pos_px = n_pos
    rec = {
        'label': args.label or os.path.basename(args.maps.rstrip('/')),
        'maps': args.maps,
        'gt': args.gt,
        'images': len(names) - len(missing),
        'missing': missing,
        'image_shapes': [list(s) for s in sorted(shapes)],
        'n_pos_px': n_pos_px,
        'n_neg_px': n_neg,
        'pos_hist': [int(v) for v in pos_hist],
        'neg_hist': [int(v) for v in neg_hist],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)

    # quick console summary at the binary-map threshold (>=128) and full range
    for t in (128,):
        tp = int(pos_hist[t:].sum())
        fp = int(neg_hist[t:].sum())
        print(f'{rec["label"]}: images={rec["images"]} pos_px={n_pos} neg_px={n_neg} '
              f'fg_rate={n_pos / max(1, n_pos + n_neg):.6%} '
              f'TPR@{t}={tp / max(1, n_pos):.5f} FPR@{t}={fp / max(1, n_neg):.3e}')
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
