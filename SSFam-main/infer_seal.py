"""Run the published SSFam RGB or RGBT checkpoint on Seal test images."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--modality", choices=("RGB", "RGBT"), default="RGBT")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--seal-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

sys.path.insert(0, str(ROOT / "代码" / args.modality))
from SAM_Model.model import Model  # noqa: E402


def index_images(directory):
    return {path.stem: path for path in directory.iterdir() if path.is_file()}


def load_thermal(path):
    with Image.open(path) as image:
        if image.mode.startswith("I;16") or image.mode == "I":
            values = np.asarray(image, dtype=np.float32)
            lo, hi = float(values.min()), float(values.max())
            if hi > lo:
                values = (values - lo) * (255.0 / (hi - lo))
            else:
                values = np.zeros_like(values)
            return Image.fromarray(np.rint(values).astype(np.uint8)).convert("RGB")
        return image.convert("RGB")


def main():
    test_dir = args.seal_dir / "test"
    rgb = index_images(test_dir / "RGB")
    thermal = index_images(test_dir / "T") if args.modality == "RGBT" else {}
    gts = sorted((test_dir / "GT").glob("*.png"))
    missing = [gt.stem for gt in gts if gt.stem not in rgb or (args.modality == "RGBT" and gt.stem not in thermal)]
    if missing:
        raise ValueError(f"Missing RGB or T for {len(missing)} GT images: {missing[:5]}")

    cfg = argparse.Namespace(model_type="vit_l", checkpoint=None)
    model = Model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    del state
    transform_rgb = transforms.Compose((
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ))
    transform_thermal = transforms.Compose((transforms.Resize((1024, 1024)), transforms.ToTensor()))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for count, gt in enumerate(gts, 1):
            with Image.open(rgb[gt.stem]) as image:
                image_tensor = transform_rgb(image.convert("RGB")).unsqueeze(0).cuda()
            if args.modality == "RGBT":
                thermal_tensor = transform_thermal(load_thermal(thermal[gt.stem])).unsqueeze(0).cuda()
                logits, _ = model(image_tensor, thermal_tensor, None)
            else:
                logits, _ = model(image_tensor, None)
            with Image.open(gt) as target:
                size = (target.height, target.width)
            prediction = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            if not cv2.imwrite(str(args.output_dir / gt.name), np.rint(prediction * 255).astype(np.uint8)):
                raise OSError(f"Could not write {args.output_dir / gt.name}")
            if count % 100 == 0 or count == len(gts):
                print(f"{args.modality}: {count}/{len(gts)}", flush=True)


if __name__ == "__main__":
    main()
