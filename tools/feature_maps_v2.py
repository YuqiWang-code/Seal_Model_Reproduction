"""Feature-response / branch-output panels, version 2 (final Ours model).

Nine columns per sample, all drawn from `Ours/CODE` and its final weights:

    01 RGB original                     02 T original
    03 SAM raw output, RGB only         04 SAM raw output, T only
    05 CM-LoRA output, RGB (pre-BSRF)   06 CM-LoRA output, T (pre-BSRF)
    07 BSRF fused feature response      08 final prediction     09 GT

Differences from v1 (see docs/Ours_Seal实验记录.md):
  * 03-06 are the *branch outputs* (decoder probabilities from that modality
    alone), not backbone feature maps, so the columns answer "what does SAM /
    CM-LoRA output before fusion"; only 07 stays a feature map, which is what
    the caption calls BSRF.
  * every probability panel uses a fixed 0-1 colour scale, so brightness is
    comparable across columns and rows;
  * the fused feature uses one global 99th-percentile scale over all samples,
    so rows can be compared;
  * a thin green box marks the GT bbox on the two original images;
  * per-panel IoU / F1 at the project threshold are written to the manifest and
    printed, giving the figure a quantitative backbone.

    CUDA_VISIBLE_DEVICES=1 Ours/venv_ours/bin/python tools/feature_maps_v2.py \
        --out Ours/Feature_Map/v2
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
    p.add_argument("--ranking", type=Path, default=root / "Ours/Feature_Map/v1/ranking.json")
    p.add_argument("--out", type=Path, default=root / "Ours/Feature_Map/v2")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--roi-min", type=int, default=64)
    p.add_argument("--roi-margin", type=int, default=24)
    p.add_argument("--zoom", type=int, default=6)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


LABELS = [
    "RGB", "T",
    "SAM raw (RGB)", "SAM raw (T)",
    "CM-LoRA (RGB)", "CM-LoRA (T)",
    "BSRF feature", "Prediction", "GT",
]


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


def prob_bgr(prob: np.ndarray, fixed: bool = True) -> np.ndarray:
    """0-1 probability -> turbo colour; fixed scale by default."""
    value = np.clip(prob.astype(np.float32), 0.0, 1.0)
    return cv2.applyColorMap((value * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)


def feature_u8(response: np.ndarray) -> np.ndarray:
    """Per-map 2-98 percentile normalisation (shape matters, not absolute scale)."""
    lo, hi = np.percentile(response, 2.0), np.percentile(response, 98.0)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return (np.clip((response - lo) / (hi - lo), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def feature_stats(response_native: np.ndarray, mask: np.ndarray) -> dict:
    """Peak/background ratio and peak distance to the GT centroid (native px)."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = bbox(mask)
    inside = np.zeros_like(mask)
    inside[max(y0 - 2, 0):y1 + 2, max(x0 - 2, 0):x1 + 2] = True
    mu_in = float(response_native[inside].mean()) if inside.any() else 0.0
    mu_out = float(response_native[~inside].mean()) if (~inside).any() else 0.0
    peak = np.unravel_index(int(np.argmax(response_native)), response_native.shape)
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = float(np.hypot(peak[0] - cy, peak[1] - cx))
    return {"ratio": mu_in / mu_out if mu_out > 1e-9 else float("inf"),
            "peak_dist_px": dist, "peak_in_gt": bool(inside[peak])}


def scores(prob: np.ndarray, mask: np.ndarray) -> dict:
    pred = prob > 0.5
    tp = int(np.count_nonzero(pred & mask))
    fp = int(np.count_nonzero(pred & ~mask))
    fn = int(np.count_nonzero(~pred & mask))
    return {
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "px": int(np.count_nonzero(pred)),
    }


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
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    stems = [row["stem"] for row in ranking[: args.topk]]

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
    text_t = encode_visual_text(t_model, device)
    find_stage = make_find_stage(device)

    dataset = RGBTBoxDataset(args.test_dir, image_size=args.image_size, load_gt=True)
    index = {sample["stem"]: i for i, sample in enumerate(dataset.samples)}

    records = []
    fused_maps = {}
    for order, stem in enumerate(stems, 1):
        item = dataset[index[stem]]
        rgb = item["rgb"].unsqueeze(0).to(device)
        thermal = item["t"].unsqueeze(0).to(device)
        boxes = item["boxes"].to(device)
        gt_mask = item["gt_mask"]
        mask = gt_mask_np(gt_mask)
        k = int(boxes.shape[0])

        def branch_prob(model, feature_dict, text, gt):
            out = decode(model, branch_backbone(feature_dict, text), boxes, find_stage)
            return prediction_probability(merge_topk_logits(out, k), gt).float().cpu().numpy()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_rgb = base.backbone.forward_image(rgb)
            base_t = base.backbone.forward_image(thermal)
            lora_rgb = rgb_model.backbone.forward_image(rgb)
            lora_t = t_model.backbone.forward_image(thermal)
            fused = m1(lora_rgb, lora_t)
            prob_raw_rgb = branch_prob(base, base_rgb, text_base, gt_mask)
            prob_raw_t = branch_prob(base, base_t, text_base, gt_mask)
            prob_lora_rgb = branch_prob(rgb_model, lora_rgb, text_rgb, gt_mask)
            prob_lora_t = branch_prob(t_model, lora_t, text_t, gt_mask)
            prob_final = branch_prob(rgb_model, fused, text_rgb, gt_mask)
        fused_response = fused["backbone_fpn"][-1].float()[0].norm(dim=0).cpu().numpy()
        fused_maps[stem] = fused_response

        records.append({
            "order": order, "stem": stem, "size": [mask.shape[1], mask.shape[0]],
            "probs": {"raw_rgb": prob_raw_rgb, "raw_t": prob_raw_t,
                      "lora_rgb": prob_lora_rgb, "lora_t": prob_lora_t, "final": prob_final},
            "mask": mask,
        })
        print(f"[{order}/{len(stems)}] {stem} computed")

    vmax = float(np.percentile(np.concatenate([m.ravel() for m in fused_maps.values()]), 99.0))
    manifest = []
    for rec in records:
        stem, mask = rec["stem"], rec["mask"]
        height, width = mask.shape
        x0, y0, x1, y1 = bbox(mask)
        sample_dir = args.out / "panels" / f"{rec['order']:02d}_{stem}"
        (sample_dir / "full").mkdir(parents=True, exist_ok=True)
        (sample_dir / "roi").mkdir(parents=True, exist_ok=True)

        rgb_native = cv2.imread(str(dataset.samples[index[stem]]["rgb"]))
        thermo = cv2.cvtColor(thermal_native(dataset.samples[index[stem]]["t"]), cv2.COLOR_GRAY2BGR)
        for canvas in (rgb_native, thermo):
            cv2.rectangle(canvas, (max(x0 - 2, 0), max(y0 - 2, 0)),
                          (min(x1 + 2, width - 1), min(y1 + 2, height - 1)), (255, 255, 0), 1)

        probs = rec["probs"]
        panel_stats = {key: scores(value, mask) for key, value in probs.items()}
        feat_stats = feature_stats(
            cv2.resize(fused_maps[stem], (width, height), interpolation=cv2.INTER_CUBIC), mask)
        panels = {
            "01_rgb.png": rgb_native,
            "02_t.png": thermo,
            "03_raw_rgb.png": prob_bgr(probs["raw_rgb"]),
            "04_raw_t.png": prob_bgr(probs["raw_t"]),
            "05_lora_rgb.png": prob_bgr(probs["lora_rgb"]),
            "06_lora_t.png": prob_bgr(probs["lora_t"]),
            "07_bsrf.png": cv2.applyColorMap(
                feature_u8(cv2.resize(fused_maps[stem], (width, height), interpolation=cv2.INTER_CUBIC)),
                cv2.COLORMAP_TURBO),
            "08_prediction.png": prob_bgr(probs["final"]),
            "09_gt.png": cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR),
        }
        box = roi_box(mask, args.roi_min, args.roi_margin)
        for name, image in panels.items():
            if not cv2.imwrite(str(sample_dir / "full" / name), image):
                raise OSError(f"cannot write {name}")
            nearest = name.startswith("09")
            roi = crop_zoom(image, box, args.zoom, nearest=nearest)
            if not cv2.imwrite(str(sample_dir / "roi" / name), roi):
                raise OSError(f"cannot write roi {name}")
        manifest.append({
            "order": rec["order"], "stem": stem, "size": rec["size"],
            "roi": [int(v) for v in box], "labels": LABELS,
            "stats": panel_stats, "feature_stats": feat_stats,
            "panels": sorted(panels), "dir": str(sample_dir),
        })
        print(f"  final IoU={panel_stats['final']['iou']:.4f} F1={panel_stats['final']['f1']:.4f} "
              f"| raw RGB {panel_stats['raw_rgb']['iou']:.3f} T {panel_stats['raw_t']['iou']:.3f} "
              f"| +LoRA RGB {panel_stats['lora_rgb']['iou']:.3f} T {panel_stats['lora_t']['iou']:.3f}")

    # colour bars: probability (fixed 0-1) and fused feature (0-vmax, normalised)
    def colorbar(kind: str) -> np.ndarray:
        ramp = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (64, 1))
        bar = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
        return bar

    cv2.imwrite(str(args.out / "legend_prob.png"), colorbar("prob"))
    cv2.imwrite(str(args.out / "legend_feature.png"), colorbar("feat"))
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol": "probabilities: fixed 0-1 turbo; BSRF: channel-L2, per-map 2-98 percentile (shape only)",
        "labels": LABELS, "topk": args.topk,
        "source": str(args.ranking), "checkpoint": str(args.checkpoint),
        "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} samples to {args.out} (vmax={vmax:.4f})")


if __name__ == "__main__":
    main()
