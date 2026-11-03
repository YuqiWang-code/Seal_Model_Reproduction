"""Prepare the Canny edge inputs supported by the NESS-Net data loader."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.data_dir / "train_data"
    rgb_edges = source / "rgb-lab"
    depth_edges = source / "depth-lab"
    rgb_edges.mkdir(exist_ok=True)
    depth_edges.mkdir(exist_ok=True)
    resize = transforms.Resize((384, 384))
    images = sorted((source / "img").glob("*.jpg"))
    for count, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            rgb = np.asarray(resize(image.convert("RGB")))
        with Image.open(source / "depth" / f"{image_path.stem}.png") as thermal:
            depth = np.asarray(resize(thermal.convert("RGB")))
        cv2.imwrite(str(rgb_edges / f"{image_path.stem}.png"), cv2.Canny(rgb, 10, 100))
        cv2.imwrite(str(depth_edges / f"{image_path.stem}.png"), cv2.Canny(depth, 10, 100))
        if count % 100 == 0 or count == len(images):
            print(f"edges: {count}/{len(images)}", flush=True)


if __name__ == "__main__":
    main()
