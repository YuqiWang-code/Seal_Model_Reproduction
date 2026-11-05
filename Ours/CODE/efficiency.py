"""计算与 test.py 相同推理路径的 Params(M)、GFLOPs、FPS。

单张图的计算是 RGB backbone、T backbone、M1（或均值融合）和一次 Decoder。
文本「visual」只编码一次，不计入 GFLOPs 和 FPS。
GFLOPs 使用乘加次数除以 1e9，一次乘加计 1，与 thop 的论文表口径一致。
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from train import (
    ROOT,
    M1,
    build_frozen_base_model,
    build_frozen_lora_model,
    build_shared_lora,
    encode_visual_text,
    forward_sample,
    make_find_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sam_checkpoint", type=Path, default=Path("/data/ckpt_files/sam_ckpt/sam3.pt"))
    parser.add_argument("--rgb_lora", type=Path, default=ROOT / "outputs" / "RGB+LoRA" / "best" / "lora_weights.pt")
    parser.add_argument("--t_lora", type=Path, default=ROOT / "outputs" / "T+LoRA" / "best" / "lora_weights.pt")
    parser.add_argument("--lora", default=True, action=argparse.BooleanOptionalAction, help="关闭时 RGB、T、LoRA3 都不用，使用冻结的原始 SAM3")
    parser.add_argument("--lora3", default=True, action=argparse.BooleanOptionalAction, help="关闭时不挂共享 LoRA3")
    parser.add_argument("--m1", default=True, action=argparse.BooleanOptionalAction, help="关闭时用 (RGB+T)/2 代替 M1")
    parser.add_argument("--img_size", type=int, default=1008)
    parser.add_argument("--num_boxes", type=int, default=1, help="几何提示框数量，Decoder 的 query 数不随它变化")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def numel(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def lora_numel(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for name, param in module.named_parameters() if "lora" in name)


def read_flops(counter) -> int:
    getter = getattr(counter, "get_total_flops", None)
    if getter is not None:
        return int(getter())
    return int(sum(counter.get_flop_counts()["Global"].values()))


def unsupported_ops(counter) -> list[str]:
    names = getattr(counter, "unspported_ops", None)
    if names is None:
        names = getattr(counter, "unsupported_ops", ())
    return sorted(str(name) for name in names)


def measure_gflops(run) -> tuple[float, list[str]]:
    """返回 GFLOPs（乘加次数 / 1e9）和未计入的算子名。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise RuntimeError("当前 PyTorch 没有 torch.utils.flop_counter.FlopCounterMode。") from exc

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        run()
    torch.cuda.synchronize()
    # PyTorch 把一次乘加计为 2 FLOPs，这里折成论文表常用的乘加次数。
    gflops = read_flops(counter) / 2 / 1e9
    return gflops, unsupported_ops(counter)


def measure_fps(run, warmup: int, iters: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return iters / (total_ms / 1e3), total_ms / iters


def build_models(args: argparse.Namespace, device: torch.device):
    if args.lora:
        rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
        t_model = build_frozen_lora_model(args.sam_checkpoint, args.t_lora, device)
        lora3 = build_shared_lora(rgb_model, t_model).to(device).eval() if args.lora3 else None
    else:
        if args.lora3:
            print("不使用模态 LoRA，LoRA3 一并关闭。")
        rgb_model = build_frozen_base_model(args.sam_checkpoint, device)
        t_model = rgb_model
        lora3 = None
    m1 = M1().to(device).eval() if args.m1 else None
    rgb_model.eval()
    t_model.eval()
    return rgb_model, t_model, m1, lora3


def main() -> None:
    args = parse_args()
    if args.img_size < 1 or args.num_boxes < 1 or args.warmup < 0 or args.iters < 1:
        raise ValueError("img_size、num_boxes、iters 必须 >= 1，warmup 必须 >= 0。")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")
    device = torch.device("cuda")

    rgb_model, t_model, m1, lora3 = build_models(args, device)
    shared = t_model is rgb_model
    rgb_params = numel(rgb_model)
    t_params = 0 if shared else numel(t_model)
    lora3_params = numel(lora3)
    m1_params = numel(m1)
    total_params = rgb_params + t_params + lora3_params + m1_params

    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)
    rgb = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    thermal = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], device=device).repeat(args.num_boxes, 1)

    def run() -> None:
        forward_sample(rgb, thermal, boxes.clone(), rgb_model, t_model, m1, text_outputs, find_stage)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Input: RGB/T 1x3x{args.img_size}x{args.img_size}, boxes={args.num_boxes}, bf16")
    print("正在统计 GFLOPs...")
    gflops, missed = measure_gflops(run)
    if gflops <= 0:
        raise RuntimeError("GFLOPs 统计结果为 0，前向没有被算子计数器记录。")
    torch.cuda.empty_cache()
    print(f"正在测量 FPS，warmup={args.warmup}，iters={args.iters}...")
    fps, latency_ms = measure_fps(run, args.warmup, args.iters)

    print()
    print(f"{'Params(M)':>12} {'GFLOPs':>12} {'FPS':>12}")
    print(f"{total_params / 1e6:12.2f} {gflops:12.2f} {fps:12.2f}")
    print()
    print(f"单张耗时: {latency_ms:.2f} ms")
    print("GFLOPs 为乘加次数 / 1e9，一次乘加计 1。文本编码不计入 GFLOPs 和 FPS。")
    rgb_lora = lora_numel(rgb_model)
    rgb_lora_text = f"（模态 LoRA {rgb_lora / 1e6:.2f} M）" if rgb_lora else ""
    print(f"RGB 模型: {rgb_params / 1e6:.2f} M{rgb_lora_text}")
    if shared:
        print("T 模型: 与 RGB 共用，不重复计入")
    else:
        t_lora = lora_numel(t_model)
        t_lora_text = f"（模态 LoRA {t_lora / 1e6:.2f} M）" if t_lora else ""
        print(f"T 模型: {t_params / 1e6:.2f} M{t_lora_text}")
    if lora3 is None:
        print("LoRA3: 关闭")
    else:
        print(f"LoRA3: {lora3_params / 1e6:.2f} M")
    if m1 is None:
        print("M1: 关闭，融合为 (RGB+T)/2")
    else:
        print(f"M1: {m1_params / 1e6:.2f} M")
    if missed:
        preview = ", ".join(missed[:30])
        if len(missed) > 30:
            preview += f" ... 另有 {len(missed) - 30} 个"
        print(f"未计入 GFLOPs 的算子: {preview}")


if __name__ == "__main__":
    main()
