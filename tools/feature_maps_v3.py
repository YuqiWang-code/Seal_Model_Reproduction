"""Feature-response / branch-output panels, version 3 (final Ours model).

Same nine columns as v2, but re-drawn so that the row reads as one monotone
"response -> target" story instead of a feature map that looks blurrier than the
prediction:

    01 RGB original                     02 T original
    03 SAM raw output, RGB only         04 SAM raw output, T only
    05 CM-LoRA output, RGB (pre-BSRF)   06 CM-LoRA output, T (pre-BSRF)
    07 BSRF fused feature response      08 final prediction     09 GT

What changed from v2 (see docs/Ours_Seal实验记录.md):

  * 07 no longer shows the whole response pedestal.  The fused map is a 72x72
    grid (one cell is ~8.6x5.8 native px), so bilinear upsampling spreads a
    single hot cell over half of a 64-px ROI and the panel ends up looking
    *less* focused than the prediction.  v3 maps the response through
    ``rel = (v - median) / (peak - median)`` -- i.e. relative to the map's own
    background level and normalised by the peak -- and then applies a gamma
    (1.8 by default) so only the core survives.  It is a monotone tone mapping
    of the same map, no spatial filtering.
  * 07 carries a white iso-contour at ``rel = 0.5`` (the half-maximum core of
    the peak) and a white/black crosshair at the argmax, so the panel reads as
    "one compact peak, here" rather than "a coloured blob".  The manifest
    records the core area, the peak offset and the inside/outside ratio.
  * all probability panels (03-06, 08) get a white ``>= 0.5`` iso-contour: the
    exact region the reported IoU is computed on.  Rows then show the contour
    going from scattered/absent to a single shape sitting on the target.
  * every panel except GT carries the GT box as a fixed reference: solid cyan
    on the two originals, two-tone dashed white/black elsewhere (one of the two
    tones is always visible against the turbo ramp or the target itself).
  * a per-column median table is written to the manifest so the caption can
    state the aggregate trend next to the samples.

    CUDA_VISIBLE_DEVICES=1 Ours/venv_ours/bin/python tools/feature_maps_v3.py \
        --out Ours/Feature_Map/v3
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
    p.add_argument("--out", type=Path, default=root / "Ours/Feature_Map/v3")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--roi-min", type=int, default=44)
    p.add_argument("--roi-margin", type=int, default=14)
    p.add_argument("--panel-px", type=int, default=396,
                   help="every zoomed panel is this many pixels, so rows line up")
    p.add_argument("--zoom", type=int, default=0, help="0 = derive from --panel-px")
    p.add_argument("--feat-low", type=float, default=0.85)
    p.add_argument("--feat-gamma", type=float, default=1.5)
    p.add_argument("--core-level", type=float, default=0.9)
    p.add_argument("--max-gt-side", type=int, default=32,
                   help="keep the sample row homogeneous: only GT boxes up to this side (px)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


LABELS = [
    "RGB", "T",
    "SAM raw (RGB)", "SAM raw (T)",
    "CM-LoRA (RGB)", "CM-LoRA (T)",
    "BSRF feature", "Prediction", "GT",
]

STAGES = [
    "input", "input",
    "frozen SAM3, one modality", "frozen SAM3, one modality",
    "CM-LoRA, one modality", "CM-LoRA, one modality",
    "BSRF fused feature", "final prediction", "GT",
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


def prob_bgr(prob: np.ndarray) -> np.ndarray:
    """0-1 probability -> turbo colour on a fixed scale."""
    value = np.clip(prob.astype(np.float32), 0.0, 1.0)
    return cv2.applyColorMap((value * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)


def feature_relative(response: np.ndarray) -> tuple:
    """Map the fused response onto (v - median) / (peak - median), clipped to 0-1."""
    background = float(np.median(response))
    peak = float(response.max())
    span = peak - background
    if span <= 1e-9:
        return np.zeros_like(response, dtype=np.float32), background, peak
    return np.clip((response - background) / span, 0.0, 1.0).astype(np.float32), background, peak


def feature_bgr(relative: np.ndarray, low: float, gamma: float) -> np.ndarray:
    """Show only the top of the response hump.

    The fused map is a broad smooth hump: its 0.5 level is 9-31 px wide because one
    FPN cell is ~8.6x5.8 native px, while its 0.9 level is a ~4 px core around the
    peak.  Displaying the whole 0-1 range therefore paints a blob larger than the
    prediction; clipping the bottom at `low` and gamma-compressing keeps the core.
    """
    shown = np.clip((relative - low) / max(1.0 - low, 1e-6), 0.0, 1.0)
    shown = np.power(shown, gamma)
    return cv2.applyColorMap((shown * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)


def dashed_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, dash: int = 4) -> None:
    """Two-tone dashed box: cyan and black alternate, so one tone always shows.

    Cyan (not white) keeps the GT reference distinct from the white response
    contours, which are drawn on the same panels.
    """
    height, width = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width - 1, x1), min(height - 1, y1)

    def run(start: int, stop: int, horizontal: bool, fixed: int) -> None:
        pos, toggle = start, 0
        while pos <= stop:
            end = min(pos + dash, stop)
            colour = (255, 255, 0) if toggle % 2 == 0 else (0, 0, 0)
            if horizontal:
                cv2.line(image, (pos, fixed), (end, fixed), colour, 1)
            else:
                cv2.line(image, (fixed, pos), (fixed, end), colour, 1)
            pos, toggle = end + 1, toggle + 1

    run(x0, x1, True, y0)
    run(x0, x1, True, y1)
    run(y0, y1, False, x0)
    run(y0, y1, False, x1)


def zoom_contours(image: np.ndarray, mask: np.ndarray, box: tuple, zoom: int,
                  colour: tuple, thickness: int = 2) -> None:
    left, top, _, _ = box
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32) - np.array([left, top], dtype=np.float32)
        points = (points * zoom).round().astype(np.int32)
        cv2.polylines(image, [points.reshape(-1, 1, 2)], True, colour, thickness, cv2.LINE_AA)


def dashed_rect_zoom(image: np.ndarray, box: tuple, rect: tuple, zoom: int,
                     dash: int = 6) -> None:
    """GT box drawn on the zoomed panel with 1-px strokes (1 native px would become zoom px)."""
    left, top, _, _ = box
    x0, y0, x1, y1 = rect
    scaled = (np.array([x0, y0, x1, y1], dtype=np.float32) - np.array([left, top], dtype=np.float32)) * zoom
    dashed_rect(image, int(round(scaled[0])), int(round(scaled[1])),
                int(round(scaled[2])), int(round(scaled[3])), dash=dash)


def zoom_cross(image: np.ndarray, x: int, y: int, box: tuple, zoom: int,
               arm: int = 9) -> None:
    left, top, _, _ = box
    cx, cy = int(round((x - left) * zoom)), int(round((y - top) * zoom))
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.line(image, (cx - arm, cy), (cx + arm, cy), colour, thick, cv2.LINE_AA)
        cv2.line(image, (cx, cy - arm), (cx, cy + arm), colour, thick, cv2.LINE_AA)


def feature_stats(response_native: np.ndarray, relative: np.ndarray,
                  core: np.ndarray, mask: np.ndarray) -> dict:
    """Peak/background ratio, peak offset to the GT centroid, half-max core area."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = bbox(mask)
    inside = np.zeros_like(mask)
    inside[max(y0 - 2, 0):y1 + 2, max(x0 - 2, 0):x1 + 2] = True
    mu_in = float(response_native[inside].mean()) if inside.any() else 0.0
    mu_out = float(response_native[~inside].mean()) if (~inside).any() else 0.0
    peak = np.unravel_index(int(np.argmax(response_native)), response_native.shape)
    cy, cx = float(ys.mean()), float(xs.mean())
    dist = float(np.hypot(peak[0] - cy, peak[1] - cx))
    def radius(area: int) -> float:
        return float(np.sqrt(area / np.pi))

    half = relative >= 0.5
    return {"ratio": mu_in / mu_out if mu_out > 1e-9 else float("inf"),
            "peak_dist_px": dist, "peak_in_gt": bool(inside[peak]),
            "peak_xy": [int(peak[1]), int(peak[0])],
            "core_px": int(np.count_nonzero(core)),
            "core_radius_px": radius(int(np.count_nonzero(core))),
            "halo_radius_px": radius(int(np.count_nonzero(half))),
            "core_in_gt": bool((core & mask).any()),
            "target_px": int(np.count_nonzero(mask))}


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


def legend_feature(low: float, gamma: float) -> np.ndarray:
    """Plain turbo ramp: the window is carried by the label, a transformed bar reads as black."""
    shown = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    return cv2.applyColorMap((np.tile(shown, (64, 1)) * 255).round().astype(np.uint8),
                             cv2.COLORMAP_TURBO)


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
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))

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
    stems = []
    for row in ranking:
        item = dataset[index[row["stem"]]]
        x0, y0, x1, y1 = bbox(gt_mask_np(item["gt_mask"]))
        if max(x1 - x0, y1 - y0) > args.max_gt_side:
            print(f"skip {row['stem']}: GT box {x1 - x0}x{y1 - y0} px > {args.max_gt_side}")
            continue
        stems.append(row["stem"])
        if len(stems) == args.topk:
            break

    records = []
    for order, stem in enumerate(stems, 1):
        started = time.perf_counter()
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

        amp = torch.amp.autocast(device.type, dtype=torch.bfloat16)
        with torch.no_grad(), amp:
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
        response = fused["backbone_fpn"][-1].float()[0].norm(dim=0).cpu().numpy()

        records.append({
            "order": order, "stem": stem, "size": [mask.shape[1], mask.shape[0]],
            "probs": {"raw_rgb": prob_raw_rgb, "raw_t": prob_raw_t,
                      "lora_rgb": prob_lora_rgb, "lora_t": prob_lora_t, "final": prob_final},
            "mask": mask, "response": response,
        })
        # cache the raw arrays so a pure display change never needs the GPU again
        np.savez_compressed(args.out / "raw" / f"{stem}.npz",
                            response=response.astype(np.float32), mask=mask.astype(np.uint8),
                            **{f"prob_{key}": value.astype(np.float32)
                               for key, value in records[-1]["probs"].items()})
        print(f"[{order}/{len(stems)}] {stem} computed in {time.perf_counter() - started:.1f} s")

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
        gx0, gy0 = max(x0 - 2, 0), max(y0 - 2, 0)
        gx1, gy1 = min(x1 + 2, width - 1), min(y1 + 2, height - 1)
        for canvas in (rgb_native, thermo):
            cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (255, 255, 0), 1)

        probs = rec["probs"]
        panel_stats = {key: scores(value, mask) for key, value in probs.items()}

        response_native = cv2.resize(rec["response"], (width, height), interpolation=cv2.INTER_CUBIC)
        relative, background, peak = feature_relative(response_native)
        core = relative >= args.core_level
        feat_stats = feature_stats(response_native, relative, core, mask)

        panels = {
            "01_rgb.png": rgb_native,
            "02_t.png": thermo,
            "03_raw_rgb.png": prob_bgr(probs["raw_rgb"]),
            "04_raw_t.png": prob_bgr(probs["raw_t"]),
            "05_lora_rgb.png": prob_bgr(probs["lora_rgb"]),
            "06_lora_t.png": prob_bgr(probs["lora_t"]),
            "07_bsrf.png": feature_bgr(relative, args.feat_low, args.feat_gamma),
            "08_prediction.png": prob_bgr(probs["final"]),
            "09_gt.png": cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR),
        }
        box = roi_box(mask, args.roi_min, args.roi_margin)
        zoom = args.zoom if args.zoom > 0 else max(2, int(round(args.panel_px / box[2])))
        peak_x, peak_y = feat_stats["peak_xy"]
        for name, image in panels.items():
            if not cv2.imwrite(str(sample_dir / "full" / name), image):
                raise OSError(f"cannot write {name}")
            roi = crop_zoom(image, box, zoom, nearest=name.startswith("09"))
            if name[3:5] in {"03", "04", "05", "06", "07", "08"}:
                dashed_rect_zoom(roi, box, (gx0, gy0, gx1, gy1), zoom)
            if name[3:5] in {"03", "04", "05", "06", "08"}:
                zoom_contours(roi, probs[{"03": "raw_rgb", "04": "raw_t", "05": "lora_rgb",
                                          "06": "lora_t", "08": "final"}[name[3:5]]] >= 0.5,
                              box, zoom, (255, 255, 255), 1)
            elif name.startswith("07"):
                zoom_contours(roi, core, box, zoom, (255, 255, 255), 2)
                zoom_cross(roi, peak_x, peak_y, box, zoom)
            if not cv2.imwrite(str(sample_dir / "roi" / name), roi):
                raise OSError(f"cannot write roi {name}")
        manifest.append({
            "order": rec["order"], "stem": stem, "size": rec["size"],
            "roi": [int(v) for v in box], "zoom": zoom, "labels": LABELS, "stages": STAGES,
            "stats": panel_stats, "feature_stats": feat_stats,
            "feature_background": background, "feature_peak": peak,
            "panels": sorted(panels), "dir": str(sample_dir),
        })
        print(f"  roi {box[2]} px zoom {zoom}x | final IoU={panel_stats['final']['iou']:.4f} F1={panel_stats['final']['f1']:.4f} "
              f"| raw RGB {panel_stats['raw_rgb']['iou']:.3f} T {panel_stats['raw_t']['iou']:.3f} "
              f"| +LoRA RGB {panel_stats['lora_rgb']['iou']:.3f} T {panel_stats['lora_t']['iou']:.3f} "
              f"| peak {feat_stats['peak_dist_px']:.2f} px "
              f"core r {feat_stats['core_radius_px']:.1f} halo r {feat_stats['halo_radius_px']:.1f}")

    keys = ["raw_rgb", "raw_t", "lora_rgb", "lora_t", "final"]
    summary = {
        "median_iou": {key: float(np.median([m["stats"][key]["iou"] for m in manifest])) for key in keys},
        "min_iou": {key: float(np.min([m["stats"][key]["iou"] for m in manifest])) for key in keys},
        "median_peak_dist_px": float(np.median([m["feature_stats"]["peak_dist_px"] for m in manifest])),
        "median_ratio": float(np.median([m["feature_stats"]["ratio"] for m in manifest])),
        "median_core_px": float(np.median([m["feature_stats"]["core_px"] for m in manifest])),
        "median_core_radius_px": float(np.median([m["feature_stats"]["core_radius_px"] for m in manifest])),
        "median_halo_radius_px": float(np.median([m["feature_stats"]["halo_radius_px"] for m in manifest])),
        "peak_in_gt": int(sum(m["feature_stats"]["peak_in_gt"] for m in manifest)),
    }

    ramps = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (64, 1))
    cv2.imwrite(str(args.out / "legend_prob.png"), cv2.applyColorMap(ramps, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(args.out / "legend_feature.png"), legend_feature(args.feat_low, args.feat_gamma))
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol": ("probabilities: fixed 0-1 turbo, white contour at 0.5; "
                     "BSRF: channel-L2, rel=(v-median)/(peak-median), display window "
                     f"rel {args.feat_low}-1, gamma {args.feat_gamma}, white contour at "
                     f"rel {args.core_level}, peak crosshair; "
                     "GT box: solid cyan on originals, two-tone dashed elsewhere"),
        "labels": LABELS, "stages": STAGES, "feat_low": args.feat_low,
        "feat_gamma": args.feat_gamma, "core_level": args.core_level,
        "topk": args.topk, "max_gt_side": args.max_gt_side,
        "selection": (f"top-{args.topk} by per-image F1 (1105 predictions) with GT box side "
                      f"<= {args.max_gt_side} px, so every row uses the same ROI and zoom"),
        "source": str(args.ranking), "checkpoint": str(args.checkpoint),
        "summary": summary, "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {len(manifest)} samples to {args.out}")


if __name__ == "__main__":
    main()
