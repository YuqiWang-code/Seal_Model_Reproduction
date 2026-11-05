"""复核 Ours/CODE 的 Params / GFLOPs / FPS：与 efficiency.py 同一推理路径，多轮取中位数。

在服务器 Ours/CODE 下运行：
  CUDA_VISIBLE_DEVICES=1 python /tmp/verify_eff.py --sam_checkpoint .../sam3.pt
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch

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


def numel(module):
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def count_flops(run):
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        run()
    torch.cuda.synchronize()
    total = int(sum(counter.get_flop_counts()["Global"].values()))
    missed = sorted(str(n) for n in getattr(counter, "unspported_ops", ()))
    return total / 2 / 1e9, missed


def heat(device, seconds=4.0):
    """先把 GPU 时钟拉满：大矩阵乘连续跑 seconds 秒。"""
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        a @ b
    torch.cuda.synchronize()


def timed(run, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            run()
            ends[i].record()
        torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_checkpoint", type=Path, default=Path("/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt"))
    ap.add_argument("--rgb_lora", type=Path, default=ROOT / "outputs" / "RGB+LoRA" / "best" / "lora_weights.pt")
    ap.add_argument("--t_lora", type=Path, default=ROOT / "outputs" / "T+LoRA" / "best" / "lora_weights.pt")
    ap.add_argument("--img_size", type=int, default=1008)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda")
    rgb_model = build_frozen_lora_model(args.sam_checkpoint, args.rgb_lora, device)
    t_model = build_frozen_lora_model(args.sam_checkpoint, args.t_lora, device)
    lora3 = build_shared_lora(rgb_model, t_model).to(device).eval()
    m1 = M1().to(device).eval()

    rgb_params, t_params = numel(rgb_model), numel(t_model)
    total = rgb_params + t_params + numel(lora3) + numel(m1)
    trainable = sum(p.numel() for p in list(m1.parameters()) + list(lora3.parameters()))

    text_outputs = encode_visual_text(rgb_model, device)
    find_stage = make_find_stage(device)
    rgb = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    thermal = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], device=device)

    def full():
        forward_sample(rgb, thermal, boxes.clone(), rgb_model, t_model, m1, text_outputs, find_stage)

    def backbones():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            rgb_model.backbone.forward_image(rgb)
            t_model.backbone.forward_image(thermal)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Params: total {total / 1e6:.3f} M | RGB {rgb_params / 1e6:.3f} | T {t_params / 1e6:.3f} "
          f"| LoRA3 {numel(lora3) / 1e6:.3f} | M1 {numel(m1) / 1e6:.3f} | trainable {trainable / 1e6:.3f}")

    gflops_full, missed_full = count_flops(full)
    gflops_bb, missed_bb = count_flops(backbones)
    print(f"GFLOPs(MAdds) full {gflops_full:.3f} | backbones {gflops_bb:.3f} | decoder+M1 {gflops_full - gflops_bb:.3f}")
    if missed_full:
        print(f"  FlopCounter 未计: {', '.join(missed_full[:10])}")

    heat(device)
    best = None
    for r in range(args.rounds):
        lat = timed(full, args.warmup, args.iters)
        med, mean, p10, p90 = statistics.median(lat), statistics.fmean(lat), *statistics.quantiles(lat, n=10)[::8]
        print(f"  round {r + 1}: median {med:.2f} ms  mean {mean:.2f}  p10 {p10:.2f}  p90 {p90:.2f}  fps {1000 / med:.2f}")
        if best is None or med < best[0]:
            best = (med, mean)
    print(f"BEST median {best[0]:.2f} ms -> {1000 / best[0]:.2f} FPS | mean {best[1]:.2f} ms -> {1000 / best[1]:.2f} FPS")


if __name__ == "__main__":
    main()
