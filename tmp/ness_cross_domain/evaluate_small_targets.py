"""Evaluate saved grayscale predictions with the shared AS1000Metrics protocol."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from metrics import AS1000Metrics


def evaluate_small_targets(pred_dir, gt_dir):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    gt_files = sorted(gt_dir.glob('*.png'))
    pred_names = {p.name for p in pred_dir.glob('*.png')}
    gt_names = {p.name for p in gt_files}
    if not gt_files or pred_names != gt_names:
        raise ValueError(f'PNG pairing mismatch: {len(gt_names - pred_names)} missing, '
                         f'{len(pred_names - gt_names)} extra; {len(gt_files)} GT images')
    accumulator = AS1000Metrics()
    for index, gt_path in enumerate(gt_files, 1):
        pred_path = pred_dir / gt_path.name
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            raise ValueError(f'Cannot decode prediction/GT pair: {gt_path.name}')
        if pred.shape != gt.shape:
            raise ValueError(f'Native-resolution shape mismatch: {gt_path.name}')
        accumulator.update(pred.astype(np.float64) / 255.0, gt > 128)
        if index % 500 == 0:
            print(f'{index}/{len(gt_files)}', flush=True)
    return {
        'metrics': accumulator.get(),
        'counts': {name: int(getattr(accumulator, name)) for name in [
            'n_images', 'tp', 'fp', 'fn', 'n_gt', 'n_pred', 'n_detected',
            'false_alarm_pixels', 'total_pixels']},
        'protocol': {
            'name': 'AS1000Metrics', 'threshold': 0.5, 'threshold_operator': '>',
            'gt_threshold_uint8': 128, 'match_distance': 3,
            'distance_operator': '<', 'connectivity': 8,
            'matching': 'raster-order greedy one-to-one centroid matching',
            'aggregation': 'whole-dataset counts',
            'Fa_definition': 'unmatched prediction component area / total image pixels',
            'units': 'raw ratios [0,1]',
        },
        'prediction_dir': str(pred_dir.resolve()), 'gt_dir': str(gt_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred_root', type=Path, required=True)
    parser.add_argument('--gt_dir', type=Path, required=True)
    parser.add_argument('--json_output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_small_targets(args.pred_root, args.gt_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
