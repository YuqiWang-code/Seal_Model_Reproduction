"""Generate MIR-Net training masks from Seal RGB images and box prompts."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--sam_checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image_dir = args.data_root / 'train' / 'RGB'
    box_dir = args.data_root / 'train' / 'box'
    paths = sorted(image_dir.glob('*.jpg'))
    model = sam_model_registry['vit_h'](checkpoint=str(args.sam_checkpoint)).cuda().eval()
    predictor = SamPredictor(model)

    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            predictor.set_image(image)
            result = np.zeros((args.size, args.size), dtype=np.uint8)
            for line in (box_dir / f'{path.stem}.txt').read_text().splitlines():
                x1, y1, x2, y2 = map(int, line.split())
                box = np.array([max(0, x1), max(0, y1),
                                min(width - 1, x2), min(height - 1, y2)], dtype=np.float32)
                masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                mask = cv2.resize(masks[np.argmax(scores)].astype(np.uint8),
                                  (args.size, args.size), interpolation=cv2.INTER_NEAREST)
                left = max(0, int(np.floor(box[0] * args.size / width)))
                top = max(0, int(np.floor(box[1] * args.size / height)))
                right = min(args.size, int(np.ceil((box[2] + 1) * args.size / width)))
                bottom = min(args.size, int(np.ceil((box[3] + 1) * args.size / height)))
                inside_box = np.zeros_like(mask)
                inside_box[top:bottom, left:right] = mask[top:bottom, left:right]
                mask = inside_box
                if not mask.any():
                    cx = min(args.size - 1, int((box[0] + box[2]) * args.size / (2 * width)))
                    cy = min(args.size - 1, int((box[1] + box[3]) * args.size / (2 * height)))
                    mask[cy, cx] = 1
                result |= mask
            cv2.imwrite(str(args.output / f'{path.stem}.png'), result * 255)
            if index % 50 == 0 or index == len(paths):
                print(f'SAM box masks {index}/{len(paths)}', flush=True)


if __name__ == '__main__':
    main()
