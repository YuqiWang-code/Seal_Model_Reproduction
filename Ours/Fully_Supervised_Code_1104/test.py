"""用 M1 消融权重导出测试集预测图。默认不加载任何 LoRA。"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAL_CUDA", "1"))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import get_loader
from train import (
    ROOT,
    M1,
    build_frozen_base_model,
    build_frozen_lora_model,
    build_shared_lora,
    encode_visual_text,
    forward_sample,
    make_find_stage,
    prediction_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/best.pth"), help="train.py 保存的 M1 权重")
    parser.add_argument("--sam_checkpoint", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt"))
    parser.add_argument("--rgb_lora", type=Path, default=ROOT / "outputs" / "RGB+LoRA" / "best" / "lora_weights.pt")
    parser.add_argument("--t_lora", type=Path, default=ROOT / "outputs" / "T+LoRA" / "best" / "lora_weights.pt")
    parser.add_argument("--lora", default=False, action=argparse.BooleanOptionalAction, help="本包默认关闭。打开后才会加载 RGB LoRA、T LoRA 和 checkpoint 里的 LoRA3")
    parser.add_argument("--m1", default=True, action=argparse.BooleanOptionalAction, help="关闭时用 (RGB+T)/2 代替 M1")
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--save_dir", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps"))
    parser.add_argument("--test_data_dir", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/Seal_data/test"), help="含 RGB/T/box/GT 的测试集根目录")
    return parser.parse_args()


def load_m1(state: dict, device: torch.device) -> M1 | None:
    m1_state = state.get("m1")
    if m1_state is None:
        print("Checkpoint 不含 M1，使用 (RGB+T)/2")
        return None
    m1 = M1().to(device)
    m1.load_state_dict(m1_state)
    m1.eval()
    return m1


def load_lora3(state: dict, rgb_model, t_model, device: torch.device) -> None:
    lora3_state = state.get("lora3")
    if not lora3_state:
        return
    lora3 = build_shared_lora(rgb_model, t_model)
    lora3.load_state_dict(lora3_state)
    lora3.to(device)
    lora3.eval()
    print("Loaded LoRA3")


@torch.no_grad()
def infer_dataset(
    rgb_model,
    t_model,
    m1: M1 | None,
    text_outputs: dict,
    find_stage,
    dataset_root: Path,
    out_dir: Path,
    img_size: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = get_loader(
        dataset_root,
        image_size=img_size,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        load_gt=True,
    )
    print(f"    Dataset: {dataset_root.name} | Samples: {len(loader)}")

    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
        thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
        gt_boxes = batch["boxes"].cuda(non_blocking=True)
        gt_mask = batch["gt_mask"]
        pred_logit = forward_sample(rgb, thermal, gt_boxes, rgb_model, t_model, m1, text_outputs, find_stage)
        probability = prediction_probability(pred_logit, gt_mask).cpu().numpy()
        image = (probability * 255).round().astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{batch['stem']}.png"), image)
    return len(loader)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")
    device = torch.device("cuda")

    print("\n=== Testing checkpoint ===")
    print(f"Checkpoint: {args.checkpoint}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        state = {"m1": state}

    if args.lora:
        rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
        t_model = build_frozen_lora_model(args.sam_checkpoint, args.t_lora, device)
        load_lora3(state, rgb_model, t_model, device)
    else:
        print("不使用 LoRA，RGB 与 T 共用冻结的原始 SAM3")
        rgb_model = build_frozen_base_model(args.sam_checkpoint, device)
        t_model = rgb_model

    if args.m1:
        m1 = load_m1(state, device)
        if m1 is None:
            raise RuntimeError("当前要求加载 M1，但 checkpoint 里没有 M1 权重。测试该消融请加 --no-m1。")
    else:
        print("不使用 M1，融合改为 (RGB+T)/2")
        m1 = None
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    save_root = args.save_dir
    print("  -> Inferring Seal test...")
    count = infer_dataset(
        rgb_model,
        t_model,
        m1,
        text_outputs,
        find_stage,
        args.test_data_dir,
        save_root,
        args.img_size,
    )
    print(f"  -> Saved {count} maps to: {save_root}")
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
