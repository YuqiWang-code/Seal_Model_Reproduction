#!/usr/bin/env python3
"""Build paper-ready qualitative panels for the Seal comparison figure.

For every requested method the script reads the binary prediction maps produced by
`tools/make_binary_maps.py`, crops a window around each sample's GT bounding box,
zooms it, and writes one PNG per sample plus the per-method crops. Column layout
of each panel: RGB (optional) | Thermal | GT | prediction overlay
(green = GT, blue = predicted binary mask).

Usage:
    python tools/build_qualitative_panels.py --samples docs/Seal定性对比样例.json \
        --methods 'DNANet=results/DNANet/Seal/test_binary' ... --out qualitative
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEAL = Path('/share_datasets/Seal_Model_Reproduction')
MARGIN = 24


def crop_window(gt, margin=MARGIN):
    ys, xs = np.where(gt > 127)
    if len(ys) == 0:
        h, w = gt.shape
        return 0, min(h, 128), 0, min(w, 128)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(gt.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(gt.shape[1], int(xs.max()) + margin + 1)
    return y0, y1, x0, x1


def pick_window(gt, max_side=112):
    """Crop around the GT box when it is compact, otherwise use the full frame."""
    y0, y1, x0, x1 = crop_window(gt)
    if (y1 - y0) <= max_side and (x1 - x0) <= max_side:
        return y0, y1, x0, x1
    h, w = gt.shape
    return 0, h, 0, w


def pick_zoom(window, target=560):
    y0, y1, x0, x1 = window
    side = max(y1 - y0, x1 - x0)
    return max(1, min(8, int(round(target / max(side, 1)))))


def zoom(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def overlay(binary, gt, window, factor):
    y0, y1, x0, x1 = window
    base = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    base[..., 0] = binary[y0:y1, x0:x1]
    base[..., 1] = np.maximum(base[..., 1], gt[y0:y1, x0:x1])
    return zoom(base, factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--methods', nargs='+', required=True, help='Name=dir pairs')
    ap.add_argument('--out', default=str(SEAL / 'qualitative'))
    ap.add_argument('--zoom', type=int, default=6)
    args = ap.parse_args()
    factor = args.zoom

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = [m.split('=', 1) for m in args.methods]
    samples = json.loads(Path(args.samples).read_text(encoding='utf-8'))['samples']

    for sample in samples:
        stem = sample['id']
        gt = cv2.imread(str(SEAL / 'Seal_data/test/GT' / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
        thermal = cv2.imread(str(SEAL / 'Seal_data/test/T' / f'{stem}.tif'), cv2.IMREAD_UNCHANGED)
        if thermal is None or gt is None:
            raise RuntimeError(f'missing inputs for {stem}')
        thermal = cv2.normalize(thermal.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb = cv2.imread(str(SEAL / 'Seal_data/test/RGB' / f'{stem}.jpg'))
        window = pick_window(gt)
        factor = pick_zoom(window, target=560)
        y0, y1, x0, x1 = window

        head = []
        if rgb is not None:
            head.append(zoom(rgb[y0:y1, x0:x1], factor))
        head.append(zoom(cv2.cvtColor(thermal[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), factor))
        head.append(zoom(cv2.cvtColor(gt[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), factor))
        sep = (60, 60, 60)
        head = [cv2.copyMakeBorder(c, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=sep) for c in head]

        rows = []
        for name, pred_dir in methods:
            binary = cv2.imread(str(Path(pred_dir) / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
            if binary is None:
                raise RuntimeError(f'missing prediction {pred_dir}/{stem}.png')
            row = head + [overlay(binary, gt, window, factor)]
            rows.append(np.hstack(row))

        panel = np.vstack(rows)
        sample_dir = out_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / 'panel.png'), panel)
        cv2.imwrite(str(sample_dir / 'thermal.png'), zoom(cv2.cvtColor(thermal[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), factor))
        cv2.imwrite(str(sample_dir / 'gt.png'), zoom(cv2.cvtColor(gt[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), factor))
        for name, pred_dir in methods:
            binary = cv2.imread(str(Path(pred_dir) / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(sample_dir / f'{name}.png'), zoom(cv2.cvtColor(binary[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), factor))
        print(f'{stem}: panel {panel.shape}, gt_px {int((gt > 127).sum())}, methods {len(methods)}')

    print(f'panels written to {out_root}')


if __name__ == '__main__':
    main()
