"""Compare aggregation choices for the Ours feature-response maps.

For each of the 10 samples selected by per-image F1 (`ranking.json` from v1) it
computes, for five backbone/fusion feature maps, four candidate aggregations and
reports how well the response localises the GT target.

Aggregations
    mean   channel mean (saturates: bf16 features share a large DC component)
    l2     channel L2 norm (v1 default)
    max    channel max (strongest single channel)
    pca1   first principal component over channels, per map
    topk   mean of the 30 channels with the largest inside/outside contrast
           (uses GT -> visualisation only, always disclosed in the caption)

Localisation metrics, computed on the ROI window
    ratio   mean(response inside GT bbox +2 px) / mean(outside)
    contrast (inside - outside) / (inside + outside)
    hit     whether the ROI argmax falls inside the GT bbox
    dist    distance (native px) from the ROI argmax to the GT centroid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


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
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json-out", type=Path)
    return p.parse_args()


def aggregations(feature: torch.Tensor, gt_mask: torch.Tensor) -> dict:
    """(1, C, h, w) feature -> {name: (h, w) float array}."""
    f = feature.float()[0].detach().cpu().numpy()          # C, h, w
    flat = f.reshape(f.shape[0], -1)
    out = {
        "mean": flat.mean(axis=0),
        "l2": np.linalg.norm(flat, axis=0),
        "max": flat.max(axis=0),
    }
    centred = flat - flat.mean(axis=1, keepdims=True)
    # first principal component over pixels: channel direction u[:, 0]
    u, sv, _ = np.linalg.svd(centred, full_matrices=False)
    comp = u[:, 0]
    if comp.sum() < 0:
        comp = -comp
    out["pca1"] = centred.T @ comp
    # topk channels by inside/outside contrast, using the GT mask
    arr = gt_mask.numpy()
    mask = arr.reshape(arr.shape[-2], arr.shape[-1]) > 0
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if ys.size:
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, height)
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, width)
        inside = np.zeros_like(mask)
        inside[y0:y1, x0:x1] = True
    else:
        inside = mask
    import cv2
    inside_small = cv2.resize(inside.astype(np.uint8), (f.shape[2], f.shape[1]),
                              interpolation=cv2.INTER_NEAREST) > 0
    inside_flat = inside_small.reshape(-1)
    if inside_flat.any() and (~inside_flat).any():
        mu_in = flat[:, inside_flat].mean(axis=1)
        mu_out = flat[:, ~inside_flat].mean(axis=1)
        score = (mu_in - mu_out) / (np.abs(mu_in) + np.abs(mu_out) + 1e-6)
        top = np.argsort(score)[-30:]
        out["topk"] = flat[top].mean(axis=0)
    shape = (f.shape[1], f.shape[2])
    return {k: v.reshape(shape) for k, v in out.items()}


def localisation(response: np.ndarray, gt_mask: torch.Tensor) -> dict:
    """Ratio / contrast / argmax hit inside GT bbox, distance to GT centroid."""
    arr = gt_mask.numpy()
    mask = arr.reshape(arr.shape[-2], arr.shape[-1]) > 0
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {}
    height, width = mask.shape
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, height)
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, width)
    inside = np.zeros_like(mask)
    inside[y0:y1, x0:x1] = True
    # response is on the ROI window; map the mask to the same window by resizing
    import cv2
    inside_small = cv2.resize(inside.astype(np.uint8), (response.shape[1], response.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    mu_in = float(response[inside_small].mean()) if inside_small.any() else 0.0
    mu_out = float(response[~inside_small].mean()) if (~inside_small).any() else 0.0
    ratio = mu_in / mu_out if mu_out > 1e-9 else float("inf")
    contrast = (mu_in - mu_out) / (mu_in + mu_out + 1e-12)
    peak = np.unravel_index(int(np.argmax(response)), response.shape)
    peak_hit = bool(inside_small[peak])
    return {"ratio": ratio, "contrast": contrast, "hit": peak_hit,
            "peak": [int(peak[1]), int(peak[0])]}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.code_dir))
    from data import RGBTBoxDataset  # noqa: E402
    from BISF import M1  # noqa: E402
    from train import (  # noqa: E402
        branch_backbone, build_frozen_base_model, build_frozen_lora_model,
        build_shared_lora, decode, encode_visual_text, make_find_stage, merge_topk_logits,
    )

    device = torch.device(args.device)
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
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    dataset = RGBTBoxDataset(args.test_dir, image_size=args.image_size, load_gt=True)
    index = {sample["stem"]: i for i, sample in enumerate(dataset.samples)}

    names = ["nolora_rgb", "nolora_t", "lora_rgb", "lora_t", "bsrf"]
    rows = []
    for stem in stems:
        item = dataset[index[stem]]
        rgb = item["rgb"].unsqueeze(0).to(device)
        thermal = item["t"].unsqueeze(0).to(device)
        gt_mask = item["gt_mask"]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            b_rgb = base.backbone.forward_image(rgb)
            b_t = base.backbone.forward_image(thermal)
            l_rgb = rgb_model.backbone.forward_image(rgb)
            l_t = t_model.backbone.forward_image(thermal)
            fused = m1(l_rgb, l_t)
        features = {
            "nolora_rgb": b_rgb["backbone_fpn"][-1],
            "nolora_t": b_t["backbone_fpn"][-1],
            "lora_rgb": l_rgb["backbone_fpn"][-1],
            "lora_t": l_t["backbone_fpn"][-1],
            "bsrf": fused["backbone_fpn"][-1],
        }
        for name in names:
            aggs = aggregations(features[name], gt_mask)
            for agg_name, response in aggs.items():
                stats = localisation(response, gt_mask)
                rows.append({"stem": stem, "map": name, "agg": agg_name, **stats})

    out = Path(args.json_out) if args.json_out else None
    if out:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'map':11s} {'agg':6s} {'ratio':>7s} {'contrast':>8s} {'hit':>5s}")
    for name in names:
        for agg_name in ("mean", "l2", "max", "pca1", "topk"):
            sel = [r for r in rows if r["map"] == name and r["agg"] == agg_name and "ratio" in r]
            if not sel:
                continue
            ratio = float(np.median([r["ratio"] for r in sel]))
            contrast = float(np.median([r["contrast"] for r in sel]))
            hit = sum(bool(r["hit"]) for r in sel)
            print(f"{name:11s} {agg_name:6s} {ratio:7.2f} {contrast:8.3f} {hit:2d}/{len(sel)}")


if __name__ == "__main__":
    main()
