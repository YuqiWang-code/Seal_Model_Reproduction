"""Feature-response panels for the final Ours model (`Ours/CODE`).

Nine panels per sample, one row per sample:
    01 RGB original          02 T original
    03 w/o CM-LoRA (RGB)     04 w/o CM-LoRA (T)      (frozen SAM3 backbone response)
    05 w/ CM-LoRA (RGB)      06 w/ CM-LoRA (T)       (backbone response before BSRF)
    07 BSRF fused response
    08 final soft prediction (0-1, pseudo colour)
    09 GT mask

Samples are ranked by per-image F1 (or IoU) computed from the saved prediction
maps against the test GT, and the first `--topk` are rendered.

Server usage (from the project root, GPU 1):
    CUDA_VISIBLE_DEVICES=1 python tools/feature_maps.py \
        --out Ours/Feature_Map/v1 --topk 10 --rank-by f1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    root = Path("/share_datasets/Seal_Model_Reproduction")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-dir", type=Path, default=root / "Ours/CODE")
    p.add_argument("--sam-checkpoint", type=Path, default=root / "Ours/weights/sam3.pt")
    p.add_argument("--checkpoint", type=Path, default=root / "Ours/CODE/train_logs2/best.pth")
    p.add_argument("--rgb-lora", type=Path, default=root / "Ours/CODE/outputs/RGB+LoRA/best/lora_weights.pt")
    p.add_argument("--t-lora", type=Path, default=root / "Ours/CODE/outputs/T+LoRA/best/lora_weights.pt")
    p.add_argument("--test-dir", type=Path, default=root / "Seal_data/test")
    p.add_argument("--pred-dir", type=Path, default=root / "Ours/test_maps")
    p.add_argument("--out", type=Path, default=root / "Ours/Feature_Map/v1")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--rank-by", choices=("f1", "iou"), default="f1")
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--device", default="cuda")
    p.add_argument("--roi-min", type=int, default=64)
    p.add_argument("--roi-margin", type=int, default=24)
    p.add_argument("--zoom", type=int, default=6)
    return p.parse_args()


def image_scores(pred_dir: Path, gt_dir: Path) -> dict:
    """Per-image pixel F1 / IoU at the project's 0.5 threshold (8-bit >= 128)."""
    out = {}
    for gt_path in sorted(gt_dir.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 128
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) > 127
        if pred.shape != gt.shape:
            continue
        tp = int(np.count_nonzero(pred & gt))
        fp = int(np.count_nonzero(pred & ~gt))
        fn = int(np.count_nonzero(~pred & gt))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out[gt_path.stem] = {"f1": f1, "iou": iou}
    return out


def response_u8(feature: torch.Tensor) -> np.ndarray:
    """(1, C, h, w) feature -> 8-bit channel L2-norm response, robustly normalised."""
    f = feature.float()[0].norm(dim=0).detach().cpu().numpy()
    lo, hi = np.percentile(f, 2.0), np.percentile(f, 98.0)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def roi_box(gt_mask: torch.Tensor, min_side: int, margin: int) -> tuple:
    """Square ROI around the GT bbox, kept inside the image."""
    mask = (gt_mask.numpy() > 0)
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if ys.size == 0:
        return 0, 0, min(width, min_side), min(height, min_side)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(min_side, max(y1 - y0, x1 - x0) + 2 * margin))
    side = min(side, width, height)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = min(max(cx - side // 2, 0), width - side)
    top = min(max(cy - side // 2, 0), height - side)
    return int(left), int(top), int(side), int(side)


def crop_zoom(image: np.ndarray, box: tuple, zoom: int, nearest: bool = False) -> np.ndarray:
    left, top, side, _ = box
    crop = image[top:top + side, left:left + side]
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    return cv2.resize(crop, (side * zoom, side * zoom), interpolation=mode)


def colorize(u8: np.ndarray, size_wh: tuple) -> np.ndarray:
    big = cv2.resize(u8, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(big, cv2.COLORMAP_TURBO)


def thermal_native(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi > lo:
        values = (values - lo) * (255.0 / (hi - lo))
    else:
        values = np.zeros_like(values)
    return np.rint(values).clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.code_dir))
    from data import RGBTBoxDataset  # noqa: E402
    from BISF import M1  # noqa: E402
    from train import (  # noqa: E402
        branch_backbone, build_frozen_base_model, build_frozen_lora_model,
        build_shared_lora, decode, encode_visual_text, make_find_stage, merge_topk_logits,
    )
    from utils import prediction_probability  # noqa: E402

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    gt_dir = args.test_dir / "GT"
    scores = image_scores(args.pred_dir, gt_dir)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1][args.rank_by], kv[1]["iou"]), reverse=True)
    selected = ranked[: args.topk]
    (args.out / "ranking.json").write_text(json.dumps(
        [{"stem": s, **v, "rank": i + 1} for i, (s, v) in enumerate(ranked)], ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print("selected:", [s for s, _ in selected])

    state = torch.load(args.checkpoint, map_location="cpu")
    base = build_frozen_base_model(args.sam_checkpoint, device)
    rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
    t_model = build_frozen_lora_model(args.sam_checkpoint, args.t_lora, device)
    lora3 = build_shared_lora(rgb_model, t_model)
    lora3.load_state_dict(state["lora3"])
    lora3.to(device).eval()
    m1 = M1().to(device)
    m1.load_state_dict(state["m1"])
    m1.eval()
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    dataset = RGBTBoxDataset(args.test_dir, image_size=args.image_size, load_gt=True)
    index = {sample["stem"]: i for i, sample in enumerate(dataset.samples)}

    labels = [
        "RGB", "T",
        "w/o CM-LoRA (RGB)", "w/o CM-LoRA (T)",
        "w/ CM-LoRA (RGB)", "w/ CM-LoRA (T)",
        "BSRF fused", "Prediction", "GT",
    ]
    manifest = []
    for order, (stem, metrics) in enumerate(selected, 1):
        item = dataset[index[stem]]
        rgb = item["rgb"].unsqueeze(0).to(device)
        thermal = item["t"].unsqueeze(0).to(device)
        boxes = item["boxes"].to(device)
        gt_mask = item["gt_mask"]
        height, width = gt_mask.shape[-2:]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_rgb = base.backbone.forward_image(rgb)
            base_t = base.backbone.forward_image(thermal)
            lora_rgb = rgb_model.backbone.forward_image(rgb)
            lora_t = t_model.backbone.forward_image(thermal)
            fused = m1(lora_rgb, lora_t)
            outputs = decode(rgb_model, branch_backbone(fused, text_outputs), boxes, find_stage)
            logits = merge_topk_logits(outputs, int(boxes.shape[0]))
            probability = prediction_probability(logits, gt_mask)

        sample_dir = args.out / "panels" / f"{order:02d}_{stem}"
        (sample_dir / "full").mkdir(parents=True, exist_ok=True)
        (sample_dir / "roi").mkdir(parents=True, exist_ok=True)
        panels = {}
        rgb_native = cv2.imread(str(dataset.samples[index[stem]]["rgb"]))
        thermo = thermal_native(dataset.samples[index[stem]]["t"])
        thermo_bgr = cv2.cvtColor(thermo, cv2.COLOR_GRAY2BGR)
        panels["01_rgb.png"] = rgb_native
        panels["02_t.png"] = thermo_bgr
        panels["03_nolora_rgb.png"] = colorize(response_u8(base_rgb["backbone_fpn"][-1]), (width, height))
        panels["04_nolora_t.png"] = colorize(response_u8(base_t["backbone_fpn"][-1]), (width, height))
        panels["05_lora_rgb.png"] = colorize(response_u8(lora_rgb["backbone_fpn"][-1]), (width, height))
        panels["06_lora_t.png"] = colorize(response_u8(lora_t["backbone_fpn"][-1]), (width, height))
        panels["07_bsrf.png"] = colorize(response_u8(fused["backbone_fpn"][-1]), (width, height))
        prob_u8 = (probability.detach().float().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        panels["08_prediction.png"] = colorize(prob_u8, (width, height))
        panels["09_gt.png"] = cv2.cvtColor((gt_mask.numpy() > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        box = roi_box(gt_mask, args.roi_min, args.roi_margin)
        for name, image in panels.items():
            if not cv2.imwrite(str(sample_dir / "full" / name), image):
                raise OSError(f"cannot write {sample_dir / 'full' / name}")
            nearest = name.startswith("09")
            roi = crop_zoom(image, box, args.zoom, nearest=nearest)
            if not cv2.imwrite(str(sample_dir / "roi" / name), roi):
                raise OSError(f"cannot write {sample_dir / 'roi' / name}")

        row = {
            "order": order, "stem": stem, "f1": metrics["f1"], "iou": metrics["iou"],
            "size": [int(width), int(height)], "roi": [int(v) for v in box], "labels": labels,
            "panels": sorted(panels), "dir": str(sample_dir),
        }
        manifest.append(row)
        print(f"[{order}/{len(selected)}] {stem} F1={metrics['f1']:.5f} IoU={metrics['iou']:.5f}")

    (args.out / "manifest.json").write_text(json.dumps({
        "rank_by": args.rank_by, "topk": args.topk, "source_pred": str(args.pred_dir),
        "checkpoint": str(args.checkpoint), "labels": labels, "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} samples to {args.out}")


if __name__ == "__main__":
    main()
