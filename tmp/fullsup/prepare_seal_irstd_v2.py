"""Rebuild Seal_IRSTD with a plain per-image min-max stretch of the 16-bit thermal band.

The 1-99 percentile stretch used first amplified background noise (background std
68/255) and reduced target SNR to ~1.5; a plain min-max keeps the native 16-bit
contrast (target background difference 4-30x the background std).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEAL = Path('/share_datasets/Seal_Model_Reproduction/Seal_data')
OUT = Path('/share_datasets/Seal_Model_Reproduction/Seal_IRSTD')


def thermal_to_uint8(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f'cannot read {path}')
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-6)
    img = np.clip(img * 255.0, 0, 255)
    return np.round(img).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modality', default='T', choices=['T', 'RGB'])
    args = ap.parse_args()

    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    lists = {}
    for split in ['train', 'val', 'test']:
        src_dir = SEAL / split / ('T' if args.modality == 'T' else 'RGB')
        ext = '.tif' if args.modality == 'T' else '.jpg'
        stems = sorted(p.stem for p in src_dir.glob('*' + ext))
        for stem in stems:
            if args.modality == 'T':
                g = thermal_to_uint8(src_dir / f'{stem}.tif')
            else:
                g = cv2.imread(str(src_dir / f'{stem}.jpg'), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(OUT / 'images' / f'{stem}.png'), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            gt = cv2.imread(str(SEAL / split / 'GT' / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(OUT / 'masks' / f'{stem}.png'), ((gt > 127) * 255).astype(np.uint8))
        lists[split] = stems
        print(split, len(stems), flush=True)

    (OUT / 'trainval.txt').write_text('\n'.join(lists['train']) + '\n', encoding='utf-8')
    (OUT / 'val.txt').write_text('\n'.join(lists['val']) + '\n', encoding='utf-8')
    (OUT / 'test.txt').write_text('\n'.join(lists['test']) + '\n', encoding='utf-8')
    meta = {
        'source': str(SEAL),
        'modality': args.modality,
        'normalization': 'per-image min-max over the full 16-bit range -> uint8',
        'counts': {k: len(v) for k, v in lists.items()},
    }
    (OUT / 'dataset_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
