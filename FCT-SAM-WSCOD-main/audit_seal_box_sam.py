"""Audit fixed box-prompt labels; dense train GT is used only for diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_seal_box_sam import selected_files, read_boxes, box_union, validate_box_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int)
    args = parser.parse_args()
    files = selected_files(args.data, args.sample_count)
    invalid = validate_box_labels(args.data, args.labels, files)
    if invalid:
        raise ValueError(f'Box label audit failed: {invalid[:5]}')
    rows = []; tp = fp = fn = 0
    for path in files:
        mask = np.asarray(Image.open(args.labels / (path.stem + '.png'))) > 128
        gt = np.asarray(Image.open(args.data / 'train/GT' / (path.stem + '.png'))) > 128
        height, width = gt.shape
        boxes = read_boxes(args.data / 'train/box' / (path.stem + '.txt'), width, height)
        union = box_union(boxes, height, width)
        inter = int((mask & gt).sum()); extra = int((mask & ~gt).sum()); missed = int((~mask & gt).sum())
        tp += inter; fp += extra; fn += missed
        rows.append({'name': path.stem, 'mask_area': float(mask.mean()), 'gt_area_diagnostic': float(gt.mean()),
                     'box_area': float(union.mean()), 'iou_diagnostic': inter / max(inter + extra + missed, 1),
                     'box_fill': float(mask.sum() / union.sum())})
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'supervision': 'Box', 'n_images': len(files), 'invalid': invalid,
              'all_boxes_nonempty': True, 'outside_box_pixels': 0, 'generation_uses_gt_masks': False,
              'dense_gt_role': 'post-generation quality diagnosis only; no candidate or parameter selection',
              'mean_mask_area': float(np.mean([r['mask_area'] for r in rows])),
              'mean_gt_area_diagnostic': float(np.mean([r['gt_area_diagnostic'] for r in rows])),
              'mean_box_area': float(np.mean([r['box_area'] for r in rows])),
              'mean_iou_diagnostic': float(np.mean([r['iou_diagnostic'] for r in rows])),
              'global_iou_diagnostic': tp / max(tp + fp + fn, 1),
              'precision_diagnostic': tp / max(tp + fp, 1), 'recall_diagnostic': tp / max(tp + fn, 1),
              'images': rows}
    (args.output / 'quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    chosen = selected_files(args.data, 16)[:8]
    canvas = Image.new('RGB', (960, 8 * 160), 'white'); draw = ImageDraw.Draw(canvas)
    for row, path in enumerate(chosen):
        if not (args.labels / (path.stem + '.png')).exists():
            continue
        image = Image.open(path).convert('RGB'); width, height = image.size
        box = read_boxes(args.data / 'train/box' / (path.stem + '.txt'), width, height)[0]
        x1, y1, x2, y2 = box; padding = 10
        crop = (max(0, x1-padding), max(0, y1-padding), min(width, x2+padding+1), min(height, y2+padding+1))
        assets = [image, Image.open(args.labels/(path.stem+'.png')).convert('RGB'),
                  Image.open(args.data/'train/GT'/(path.stem+'.png')).convert('RGB')]
        for col, (asset, label) in enumerate(zip(assets, ['RGB around supplied box', 'Box-SAM pseudo', 'GT: diagnosis only'])):
            # Magnify the same box crop to inspect small objects.
            tile = asset.crop(crop).resize((260, 130), Image.Resampling.NEAREST)
            canvas.paste(tile, (col*320, row*160+25));draw.text((col*320+2,row*160+5),f'{path.stem}: {label}',fill='black')
    canvas.save(args.output / 'quality_samples.png')
    print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2),flush=True)


if __name__ == '__main__':
    main()
