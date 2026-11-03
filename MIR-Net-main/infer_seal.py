"""Run the published MIR-Net checkpoint on a Seal split."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from net import Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Seal split directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trainsize', type=int, default=384)
    args = parser.parse_args()
    args.mode = 'test'
    args.pretrained_backbone = None

    rgb_dir = args.data / 'RGB'
    gt_dir = args.data / 'GT'
    gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}
    images = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    args.output.mkdir(parents=True, exist_ok=True)

    model = Net(args).cuda().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 256
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 256

    with torch.inference_mode():
        for index, image_path in enumerate(images, 1):
            gt_path = gt_by_stem[image_path.stem]
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image - mean) / std
            image = cv2.resize(image, (args.trainsize, args.trainsize))
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
            logits, _ = model(tensor, shape=gt.shape)
            prediction = torch.sigmoid(logits).squeeze().cpu().numpy()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            cv2.imwrite(str(args.output / gt_path.name), (prediction * 255).astype(np.uint8))
            if index % 100 == 0 or index == len(images):
                print(f'{index}/{len(images)}', flush=True)


if __name__ == '__main__':
    main()
