"""Evaluate a published NESS-Net checkpoint on Seal RGB and thermal pairs."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.NESS_Net import NESS_Net


def index_images(directory):
    return {path.stem: path for path in directory.iterdir() if path.is_file()}


def index_edges(directory):
    return {path.stem.split("_Hier_")[0]: path for path in directory.glob("*.png")}


def load_thermal(path):
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high > low:
        values = (values - low) * (255.0 / (high - low))
    else:
        values = np.zeros_like(values)
    return Image.fromarray(np.rint(values).astype(np.uint8)).convert("RGB")


def edge_tensor(image, edge_path, resize):
    if edge_path is None:
        edge = cv2.Canny(np.asarray(resize(image)), 10, 100)
    else:
        with Image.open(edge_path) as saved_edge:
            edge = np.array(resize(saved_edge.convert("L")), dtype=np.uint8, copy=True)
    return transforms.functional.to_tensor(edge).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("general", "refinement"), default="refinement")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--rgb-edge-dir", type=Path)
    parser.add_argument("--thermal-edge-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if (args.rgb_edge_dir is None) != (args.thermal_edge_dir is None):
        parser.error("Pass both edge directories or neither")

    test_dir = args.seal_dir / args.split
    rgb = index_images(test_dir / "RGB")
    thermal = index_images(test_dir / "T")
    gts = sorted((test_dir / "GT").glob("*.png"))
    if args.limit is not None:
        gts = gts[: args.limit]
    missing = [gt.stem for gt in gts if gt.stem not in rgb or gt.stem not in thermal]
    if missing:
        raise ValueError(f"Missing RGB or T for {len(missing)} GT images: {missing[:5]}")

    rgb_edges = index_edges(args.rgb_edge_dir) if args.rgb_edge_dir else {}
    thermal_edges = index_edges(args.thermal_edge_dir) if args.thermal_edge_dir else {}
    if args.rgb_edge_dir:
        missing = [gt.stem for gt in gts if gt.stem not in rgb_edges or gt.stem not in thermal_edges]
        if missing:
            raise ValueError(f"Missing edge maps for {len(missing)} GT images: {missing[:5]}")

    device = torch.device("cuda")
    model = NESS_Net(channel=32, general_stage=args.stage == "general")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model = model.to(device).eval()
    del state

    resize = transforms.Resize((384, 384))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for count, gt in enumerate(gts, 1):
            with Image.open(rgb[gt.stem]) as source:
                image = source.convert("RGB")
            depth = load_thermal(thermal[gt.stem])
            image_edge = edge_tensor(image, rgb_edges.get(gt.stem), resize).to(device)
            depth_edge = edge_tensor(depth, thermal_edges.get(gt.stem), resize).to(device)
            image_tensor = normalize(to_tensor(resize(image))).unsqueeze(0).to(device)
            depth_tensor = to_tensor(resize(depth)).unsqueeze(0).to(device)

            prediction = model(image_tensor, depth_tensor, image_edge, depth_edge)[2]
            with Image.open(gt) as target:
                size = (target.height, target.width)
            prediction = F.interpolate(prediction, size=size, mode="bilinear", align_corners=False)
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            output = args.output_dir / gt.name
            if not cv2.imwrite(str(output), np.rint(prediction * 255).astype(np.uint8)):
                raise OSError(f"Could not write {output}")
            if count % 100 == 0 or count == len(gts):
                print(f"{args.stage}: {count}/{len(gts)}", flush=True)


if __name__ == "__main__":
    main()
