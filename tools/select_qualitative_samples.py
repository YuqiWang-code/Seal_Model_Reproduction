#!/usr/bin/env python3
"""Pick a deterministic, area-stratified set of Seal test samples for the qualitative figure.

Samples are chosen by quantiles of the GT foreground area (plus two multi-target
frames), so the panel set spans tiny, medium and larger targets without any
dependence on model scores.

Usage:
    python tools/select_qualitative_samples.py --out docs/Seal定性对比样例.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEAL = Path('/share_datasets/Seal_Model_Reproduction')
QUANTILES = [0.03, 0.15, 0.35, 0.55, 0.7, 0.82, 0.9, 0.95, 0.98]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_gt', default=str(SEAL / 'Seal_data/test/GT'))
    ap.add_argument('--out', default='docs/Seal定性对比样例.json')
    ap.add_argument('--count', type=int, default=9)
    args = ap.parse_args()

    gt_dir = Path(args.test_gt)
    stats = []
    for p in sorted(gt_dir.glob('*.png')):
        gt = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        area = int((gt > 127).sum())
        n_comp, _ = cv2.connectedComponents((gt > 127).astype(np.uint8), connectivity=8)
        stats.append({'id': p.stem, 'area': area, 'components': n_comp - 1})

    stats.sort(key=lambda s: (s['area'], s['id']))
    picked = []
    seen = set()
    for q in QUANTILES[:args.count]:
        idx = min(len(stats) - 1, int(round(q * (len(stats) - 1))))
        s = stats[idx]
        while s['id'] in seen and idx + 1 < len(stats):
            idx += 1
            s = stats[idx]
        seen.add(s['id'])
        picked.append(s)

    multi = [s for s in stats if s['components'] >= 5 and s['id'] not in seen]
    for s in multi[::max(1, len(multi) // 2)][:2]:
        if s['id'] not in seen:
            seen.add(s['id'])
            picked.append(s)

    out = {
        'source': str(gt_dir),
        'selection': 'quantiles of GT foreground area + two multi-target frames',
        'quantiles': QUANTILES[:args.count],
        'samples': picked,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for s in picked:
        print(s)


if __name__ == '__main__':
    main()
