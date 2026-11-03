#!/usr/bin/env python3
"""Threshold test predictions into binary small-target masks.

The project-wide small-target protocol thresholds the 8-bit probability map at
0.5 (`>0.5`). Stored PNGs are 0-255, so the equivalent is `>= 128`.

Usage:
    python tools/make_binary_maps.py --pred results/DNANet/Seal/fullsup_test_maps \
        --out results/DNANet/Seal/test_binary
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def to_binary(pred_path, gt_size=None):
    img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'cannot read {pred_path}')
    if gt_size is not None and img.shape != gt_size:
        raise ValueError(f'{pred_path.name}: {img.shape} != {gt_size}')
    return np.where(img >= 128, 255, 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gt_dir', default='/share_datasets/Seal_Model_Reproduction/Seal_data/test/GT')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    pred_dir, out_dir = Path(args.pred), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir)
    gts = {p.stem: p for p in gt_dir.glob('*.png')}
    preds = sorted(pred_dir.glob('*.png'))
    missing = set(gts) - {p.stem for p in preds}
    extra = {p.stem for p in preds} - set(gts)
    if missing or extra:
        raise SystemExit(f'pairing mismatch: {len(missing)} missing, {len(extra)} extra')

    shapes = set()
    for p in preds:
        gt = cv2.imread(str(gts[p.stem]), cv2.IMREAD_GRAYSCALE)
        binmap = to_binary(p, gt.shape)
        cv2.imwrite(str(out_dir / p.name), binmap)
        shapes.add(binmap.shape)
    print(f'{out_dir}: {len(preds)} binary maps, shapes={shapes}')


if __name__ == '__main__':
    main()
