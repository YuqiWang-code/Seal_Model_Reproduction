"""Prepare sparse box supervision for NESS-Net's Seal adaptation."""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw

from infer_seal import index_images, load_thermal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.seal_dir / "train"
    thermal = index_images(source / "T")
    for name in ("img", "depth", "gt", "mask", "gray"):
        (args.output_dir / "train_data" / name).mkdir(parents=True, exist_ok=True)

    images = sorted((source / "RGB").glob("*.jpg"))
    for count, image_path in enumerate(images, 1):
        stem = image_path.stem
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size
        foreground = Image.new("L", image.size, 0)
        valid = Image.new("L", image.size, 255)
        foreground_draw = ImageDraw.Draw(foreground)
        valid_draw = ImageDraw.Draw(valid)
        boxes = []
        for line in (source / "box" / f"{stem}.txt").read_text().splitlines():
            x1, y1, x2, y2 = map(int, line.split())
            boxes.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)))
        for box in boxes:
            valid_draw.rectangle(box, fill=0)
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_x = max(1, (x2 - x1 + 1) // 4)
            half_y = max(1, (y2 - y1 + 1) // 4)
            center = (max(x1, cx - half_x), max(y1, cy - half_y),
                      min(x2, cx + half_x), min(y2, cy + half_y))
            foreground_draw.rectangle(center, fill=255)
            valid_draw.rectangle(center, fill=255)

        destination = args.output_dir / "train_data"
        os.symlink(image_path, destination / "img" / image_path.name)
        load_thermal(thermal[stem]).save(destination / "depth" / f"{stem}.png")
        foreground.save(destination / "gt" / f"{stem}.png")
        valid.save(destination / "mask" / f"{stem}.png")
        image.convert("L").save(destination / "gray" / f"{stem}.png")
        if count % 100 == 0 or count == len(images):
            print(f"train: {count}/{len(images)}", flush=True)


if __name__ == "__main__":
    main()
