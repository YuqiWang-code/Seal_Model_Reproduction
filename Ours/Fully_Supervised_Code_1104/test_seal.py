"""全监督 RGB-T 模型的 Seal test 推理。

用法（服务器，项目根目录为 /share_datasets/Seal_Model_Reproduction）：

    cd Ours/Fully_Supervised_Code_1104
    PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
    $PY test_seal.py --weights train_logs/best.pth --test_data_dir .../Seal_data/test \
        --out_dir .../results/Ours/FullySup_1104/test_maps

流程与 train.py 完全一致：冻结 SAM3 + 两套模态 LoRA，M1 融合 RGB/T 视觉特征，
用数据集的真实框作为 Decoder 的几何提示（与训练同一入口），取 top-k mask 的逐像素
最大 logit，sigmoid 后双线性恢复到 GT 尺寸并逐图 min-max，保存 8 位 PNG。
文件名与 GT 一一对应，可直接交给项目根目录的 evaluate.py / evaluate_small_targets.py。
"""

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
    M1,
    build_frozen_lora_model,
    build_shared_lora,
    encode_visual_text,
    forward_sample,
    make_find_stage,
    prediction_probability,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal test 推理（全监督 RGB-T）")
    parser.add_argument("--weights", type=str, required=True, help="train.py 保存的 best.pth（含 m1 与 lora3）")
    parser.add_argument("--checkpoint", type=str,
                        default="/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt")
    parser.add_argument("--rgb_lora", type=str,
                        default="/share_datasets/Seal_Model_Reproduction/Ours/weights/RGB+LoRA/lora_weights.pt")
    parser.add_argument("--t_lora", type=str,
                        default="/share_datasets/Seal_Model_Reproduction/Ours/weights/T+LoRA/lora_weights.pt")
    parser.add_argument("--test_data_dir", type=str,
                        default="/share_datasets/Seal_Model_Reproduction/Seal_data/test")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 张，用于冒烟测试")
    parser.add_argument("--no_lora3", action="store_true", help="与训练时 --no-lora3 对应")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")
    device = torch.device("cuda")

    rgb_model = build_frozen_lora_model(Path(args.checkpoint), Path(args.rgb_lora), device)
    t_model = build_frozen_lora_model(Path(args.checkpoint), Path(args.t_lora), device)
    lora3 = None if args.no_lora3 else build_shared_lora(rgb_model, t_model).to(device)
    m1 = M1().to(device)

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    if "m1" not in state:
        raise RuntimeError(f"{args.weights} 里没有 m1 权重")
    m1.load_state_dict(state["m1"])
    if lora3 is not None:
        if state.get("lora3") is None:
            raise RuntimeError("权重里没有 lora3，但推理需要它（或加 --no_lora3）")
        lora3.load_state_dict(state["lora3"])
    m1.eval()
    print(f"loaded {args.weights}: epoch={state.get('epoch')} val_iou={state.get('iou')}")

    loader = get_loader(
        args.test_data_dir,
        image_size=args.img_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        load_gt=True,
    )
    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(loader) if args.limit is None else min(args.limit, len(loader))
    written = 0
    for index, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
        thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
        gt_boxes = batch["boxes"].cuda(non_blocking=True)
        gt_mask = batch["gt_mask"].cuda(non_blocking=True)

        pred_logit = forward_sample(
            rgb, thermal, gt_boxes, rgb_model, t_model, m1, text_outputs, find_stage
        )
        probability = prediction_probability(pred_logit, gt_mask).float().clamp(0.0, 1.0)
        lo, hi = float(probability.min()), float(probability.max())
        probability = (probability - lo) / max(hi - lo, 1e-8)

        array = np.rint(probability.cpu().numpy() * 255.0).astype(np.uint8)
        name = f"{batch['stem']}.png"
        if not cv2.imwrite(str(out_dir / name), array):
            raise OSError(f"无法写入 {out_dir / name}")
        written += 1
        if index % 100 == 0 or index == total:
            print(f"{index}/{total}", flush=True)
        if index >= total:
            break

    print(f"完成：{written} 张预测写入 {out_dir}")


if __name__ == "__main__":
    main()
