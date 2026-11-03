"""Generate native-resolution Seal training masks using RGB and box prompts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def read_boxes(path, width, height):
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        x1, y1, x2, y2 = map(int, line.split())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
        if x2 < x1 or y2 < y1:
            raise ValueError(f'Invalid box in {path}')
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        raise ValueError(f'No box annotation: {path}')
    return boxes


def box_union(boxes, height, width):
    union = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        union[y1:y2 + 1, x1:x2 + 1] = True
    return union


def validate_box_labels(data, folder, files):
    invalid = []
    for path in files:
        width, height = Image.open(path).size
        boxes = read_boxes(data / 'train/box' / (path.stem + '.txt'), width, height)
        label = folder / (path.stem + '.png')
        if not label.exists():
            invalid.append({'name': path.stem, 'reason': 'missing'})
            continue
        mask = np.asarray(Image.open(label).convert('L')) > 128
        if mask.shape != (height, width):
            invalid.append({'name': path.stem, 'reason': 'shape'})
        elif (mask & ~box_union(boxes, height, width)).any():
            invalid.append({'name': path.stem, 'reason': 'outside_box'})
        elif any(not mask[y1:y2 + 1, x1:x2 + 1].any() for x1, y1, x2, y2 in boxes):
            invalid.append({'name': path.stem, 'reason': 'empty_box'})
    return invalid


def selected_files(data, sample_count=None):
    files = sorted((data / 'train/RGB').glob('*.jpg'))
    if sample_count is not None:
        files = [files[i] for i in np.linspace(0, len(files) - 1, sample_count, dtype=int)]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sam', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / 'generation.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
        'supervision': 'Box', 'inputs': ['train/RGB', 'train/box'], 'sam_checkpoint': str(args.sam),
        'model': 'SAM ViT-H', 'selection': 'highest SAM predicted IoU among nonempty box-clipped candidates',
        'postprocessing': 'intersect each mask with its supplied inclusive xyxy box; union instances',
        'uses_gt_masks': False, 'images': {}}
    files = selected_files(args.data, args.sample_count)
    pending = [path for path in files if not (args.output / (path.stem + '.png')).exists()]
    if pending:
        model = sam_model_registry['vit_h'](checkpoint=str(args.sam)).cuda().eval()
        predictor = SamPredictor(model)
        with torch.inference_mode():
            for index, path in enumerate(pending, 1):
                image = np.asarray(Image.open(path).convert('RGB'))
                height, width = image.shape[:2]
                boxes = read_boxes(args.data / 'train/box' / (path.stem + '.txt'), width, height)
                predictor.set_image(image)
                result = np.zeros((height, width), dtype=bool)
                instances = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
                    candidates = []
                    for candidate in masks:
                        clipped = np.zeros_like(result)
                        clipped[y1:y2 + 1, x1:x2 + 1] = candidate[y1:y2 + 1, x1:x2 + 1]
                        candidates.append(clipped)
                    valid = [i for i, mask in enumerate(candidates) if mask.any()]
                    if not valid:
                        raise ValueError(f'All SAM candidates empty inside supplied box: {path.name} {box}')
                    best = max(valid, key=lambda i: scores[i])
                    result |= candidates[best]
                    instances.append({'box': box, 'selected': best, 'scores': scores.tolist(),
                                      'mask_pixels': int(candidates[best].sum())})
                Image.fromarray(result.astype(np.uint8) * 255).save(args.output / (path.stem + '.png'))
                metadata['images'][path.stem] = {'shape': [height, width], 'instances': instances}
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                if index % 50 == 0 or index == len(pending):
                    print(f'SAM box labels: {index}/{len(pending)}', flush=True)
    invalid = validate_box_labels(args.data, args.output, files)
    if invalid:
        raise ValueError(f'Invalid box pseudo labels: {invalid[:5]}')
    print(f'BOX_LABELS_READY {len(files)}', flush=True)


if __name__ == '__main__':
    main()
