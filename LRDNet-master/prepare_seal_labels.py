import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Convert Seal boxes to sparse LRDNet labels")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/Seal/box_labels"))
    args = parser.parse_args()

    image_dir = args.data_root / "train" / "RGB"
    box_dir = args.data_root / "train" / "box"
    args.output.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    image_count = 0
    for image_path in sorted(image_dir.glob("*.jpg")):
        width, height = Image.open(image_path).size
        label = Image.new("L", (width, height), 2)
        draw = ImageDraw.Draw(label)
        boxes = []
        for line in (box_dir / f"{image_path.stem}.txt").read_text().splitlines():
            x1, y1, x2, y2 = map(int, line.split())
            x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
            boxes.append((x1, y1, x2, y2))
            draw.rectangle((x1, y1, x2, y2), fill=0)
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_x = max(1, (x2 - x1 + 1) // 4)
            half_y = max(1, (y2 - y1 + 1) // 4)
            draw.rectangle((max(x1, cx - half_x), max(y1, cy - half_y),
                            min(x2, cx + half_x), min(y2, cy + half_y)), fill=1)
        label.save(args.output / f"{image_path.stem}.png")
        image_count += 1
        total_boxes += len(boxes)
    print(f"Saved {image_count} labels with {total_boxes} boxes to {args.output}")


if __name__ == "__main__":
    main()
