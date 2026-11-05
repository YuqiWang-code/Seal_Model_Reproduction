"""Feature-response heat maps, version 6 -- v1 look, processed.

v1 (`tools/feature_maps.py`) is kept as the reference for the *content*: each of
03-07 is the channel-L2 response of that branch's `backbone_fpn[-1]` feature,
normalised per map to its 2-98 percentile and painted with turbo.  That is what
makes the panels look structured instead of a saturated dot.

v6 keeps that content and changes only what was noisy about v1:

  * one geometry for every row: ROI = max(44 px, GT bbox + 2x14 px), panels are
    resampled to 396 px, so rows can be compared directly (v1 mixed 64 px / 6x
    with a 182 px ROI for the one large target);
  * samples come from an explicit list (`--stems`), so the set can be picked by
    a stated rule instead of "whatever ranked first";
  * no annotation of any kind on the panels (v1 also had none) and a uniform
    396x396 panel size;
  * the per-branch response maps and the prediction are cached to
    `raw/<stem>.npz`, so a display change never needs the GPU again.

    CUDA_VISIBLE_DEVICES=1 Ours/venv_ours/bin/python tools/feature_maps_v6.py \
        --out Ours/Feature_Map/v6 --stems 2845,3343,...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

LABELS = [
    "RGB", "T",
    "w/o CM-LoRA (RGB)", "w/o CM-LoRA (T)",
    "w/ CM-LoRA (RGB)", "w/ CM-LoRA (T)",
    "BSRF fused", "Prediction", "GT",
]


def parse_args() -> argparse.Namespace:
    root = Path("/share_datasets/Seal_Model_Reproduction")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-dir", type=Path, default=root / "Ours/CODE")
    p.add_argument("--sam-checkpoint", type=Path, default=root / "Ours/weights/sam3.pt")
    p.add_argument("--checkpoint", type=Path, default=root / "Ours/CODE/train_logs2/best.pth")
    p.add_argument("--rgb-lora", type=Path, default=root / "Ours/CODE/outputs/RGB+LoRA/best/lora_weights.pt")
    p.add_argument("--t-lora", type=Path, default=root / "Ours/CODE/outputs/T+LoRA/best/lora_weights.pt")
    p.add_argument("--test-dir", type=Path, default=root / "Seal_data/test")
    p.add_argument("--out", type=Path, default=root / "Ours/Feature_Map/v6")
    p.add_argument("--stems", required=True, help="comma separated sample list")
    p.add_argument("--selection-note", default="explicit list")
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--roi-min", type=int, default=44)
    p.add_argument("--roi-margin", type=int, default=14)
    p.add_argument("--panel-px", type=int, default=396)
    p.add_argument("--resp-lo", type=float, default=2.0)
    p.add_argument("--resp-hi", type=float, default=98.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def thermal_native(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    values = (values - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(values)
    return np.rint(values).clip(0, 255).astype(np.uint8)


def gt_mask_np(gt_mask: torch.Tensor) -> np.ndarray:
    arr = gt_mask.numpy()
    return arr.reshape(arr.shape[-2], arr.shape[-1]) > 0


def bbox(mask: np.ndarray) -> tuple:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def roi_box(mask: np.ndarray, min_side: int, margin: int) -> tuple:
    height, width = mask.shape
    x0, y0, x1, y1 = bbox(mask)
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


def response_u8(feature: torch.Tensor, lo_p: float, hi_p: float) -> np.ndarray:
    """(1, C, h, w) -> uint8 channel-L2 response, per-map robust scaling (v1 rule)."""
    f = feature.float()[0].norm(dim=0).detach().cpu().numpy()
    lo, hi = float(np.percentile(f, lo_p)), float(np.percentile(f, hi_p))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def colorize(u8: np.ndarray, size_wh: tuple) -> np.ndarray:
    big = cv2.resize(u8, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(big, cv2.COLORMAP_TURBO)


def prob_bgr(prob: np.ndarray, size_wh: tuple) -> np.ndarray:
    value = np.clip(np.asarray(prob, np.float32), 0.0, 1.0)
    big = cv2.resize(value, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((big * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)


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
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    stems = [s for s in args.stems.split(",") if s]

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
    text_base = encode_visual_text(base, device)
    text_rgb = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    dataset = RGBTBoxDataset(args.test_dir, image_size=args.image_size, load_gt=True)
    index = {sample["stem"]: i for i, sample in enumerate(dataset.samples)}

    manifest = []
    for order, stem in enumerate(stems, 1):
        started = time.perf_counter()
        item = dataset[index[stem]]
        rgb = item["rgb"].unsqueeze(0).to(device)
        thermal = item["t"].unsqueeze(0).to(device)
        boxes = item["boxes"].to(device)
        gt_mask = item["gt_mask"]
        mask = gt_mask_np(gt_mask)
        height, width = mask.shape
        k = int(boxes.shape[0])

        amp = torch.amp.autocast(device.type, dtype=torch.bfloat16)
        with torch.no_grad(), amp:
            base_rgb = base.backbone.forward_image(rgb)
            base_t = base.backbone.forward_image(thermal)
            lora_rgb = rgb_model.backbone.forward_image(rgb)
            lora_t = t_model.backbone.forward_image(thermal)
            fused = m1(lora_rgb, lora_t)
            outputs = decode(rgb_model, branch_backbone(fused, text_rgb), boxes, find_stage)
            probability = prediction_probability(merge_topk_logits(outputs, k), gt_mask)

        responses = {
            "nolora_rgb": response_u8(base_rgb["backbone_fpn"][-1], args.resp_lo, args.resp_hi),
            "nolora_t": response_u8(base_t["backbone_fpn"][-1], args.resp_lo, args.resp_hi),
            "lora_rgb": response_u8(lora_rgb["backbone_fpn"][-1], args.resp_lo, args.resp_hi),
            "lora_t": response_u8(lora_t["backbone_fpn"][-1], args.resp_lo, args.resp_hi),
            "bsrf": response_u8(fused["backbone_fpn"][-1], args.resp_lo, args.resp_hi),
        }
        prob = probability.detach().float().cpu().numpy()
        prob = prob.reshape(prob.shape[-2], prob.shape[-1]) if prob.ndim > 2 else prob
        np.savez_compressed(args.out / "raw" / f"{stem}.npz",
                            mask=mask.astype(np.uint8), prob=prob.astype(np.float32),
                            **{k2: v for k2, v in responses.items()})

        sample_dir = args.out / "panels" / f"{order:02d}_{stem}"
        (sample_dir / "full").mkdir(parents=True, exist_ok=True)
        (sample_dir / "roi").mkdir(parents=True, exist_ok=True)
        rgb_native = cv2.imread(str(dataset.samples[index[stem]]["rgb"]))
        thermo = cv2.cvtColor(thermal_native(dataset.samples[index[stem]]["t"]), cv2.COLOR_GRAY2BGR)
        panels = {
            "01_rgb.png": rgb_native,
            "02_t.png": thermo,
            "03_nolora_rgb.png": colorize(responses["nolora_rgb"], (width, height)),
            "04_nolora_t.png": colorize(responses["nolora_t"], (width, height)),
            "05_lora_rgb.png": colorize(responses["lora_rgb"], (width, height)),
            "06_lora_t.png": colorize(responses["lora_t"], (width, height)),
            "07_bsrf.png": colorize(responses["bsrf"], (width, height)),
            "08_prediction.png": prob_bgr(prob, (width, height)),
            "09_gt.png": cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR),
        }
        box = roi_box(mask, args.roi_min, args.roi_margin)
        zoom = max(2, int(round(args.panel_px / box[2])))
        for name, image in panels.items():
            if not cv2.imwrite(str(sample_dir / "full" / name), image):
                raise OSError(f"cannot write {name}")
            roi = crop_zoom(image, box, zoom, nearest=name.startswith("09"))
            if not cv2.imwrite(str(sample_dir / "roi" / name), roi):
                raise OSError(f"cannot write roi {name}")

        pred = prob > 0.5
        tp = int(np.count_nonzero(pred & mask))
        fp = int(np.count_nonzero(pred & ~mask))
        fn = int(np.count_nonzero(~pred & mask))
        manifest.append({
            "order": order, "stem": stem, "size": [int(width), int(height)],
            "roi": [int(v) for v in box], "zoom": zoom, "labels": LABELS,
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "gt_px": int(mask.sum()), "pred_px": int(pred.sum()),
            "panels": sorted(panels), "dir": str(sample_dir),
        })
        print(f"[{order}/{len(stems)}] {stem} roi {box[2]} px zoom {zoom}x "
              f"iou {manifest[-1]['iou']:.3f} ({time.perf_counter() - started:.1f} s)")

    ramp = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (64, 1))
    bar = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(args.out / "legend_feature.png"), bar)
    cv2.imwrite(str(args.out / "legend_prob.png"), bar)
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol": (f"03-07: channel-L2 of backbone_fpn[-1], per-map {args.resp_lo:g}-{args.resp_hi:g} "
                     "percentile scaling (v1 rule), turbo; 08: prediction, fixed 0-1 turbo; "
                     "no annotations"),
        "labels": LABELS, "selection": args.selection_note,
        "roi_min": args.roi_min, "roi_margin": args.roi_margin, "panel_px": args.panel_px,
        "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} samples to {args.out}")


if __name__ == "__main__":
    main()
