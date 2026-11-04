#!/usr/bin/env python3
"""Measure params / GFLOPs / FPS for one fully-supervised baseline.

Run with cwd = the method folder so that its own modules resolve. The model is
wrapped so that profiling sees a single tensor output (several baselines return
tuples/lists, which thop cannot trace). FLOPs use the thop convention (MACs),
which is what the IRSTD papers call "GFLOPs"; a torch FlopCounter fallback
(reported as FLOPs/2 to stay on the MACs scale) is used if thop still fails.

FPS = 1 / mean latency of a single 256x256 forward, FP32, batch 1, after warm-up,
best of 3 timing rounds to reduce interference from other processes on the GPU.

Usage:
    cd /…/DNANet-main && python /…/tools/measure_complexity_one.py --kind dnanet
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())


def build(kind):
    """Return (model, forward_fn, in_channels)."""
    if kind == 'dnanet':
        from model.model_DNANet import DNANet, Res_CBAM_block
        from model.utils import weights_init_xavier
        net = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block,
                     num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
                     deep_supervision='True')
        net.apply(weights_init_xavier)
        return net, (lambda m, x: m(x)[-1]), 3
    if kind == 'mshnet':
        from model.MSHNet import MSHNet
        return MSHNet(3), (lambda m, x: m(x, True)[1]), 3
    if kind == 'uiunet':
        from model import UIUNET
        return UIUNET(3, 1), (lambda m, x: m(x)[0]), 3
    if kind == 'sctransnet':
        from train_seal import build_model
        return build_model('test'), (lambda m, x: m(x)), 1
    if kind == 'rpcanet':
        from models import get_model
        return get_model('rpcanet'), (lambda m, x: m(x)[1]), 1
    if kind == 'hdnet':
        from model.HDNet import HDNet
        return HDNet(3), (lambda m, x: m(x, True)[1]), 3
    if kind == 'ilnet':
        from model.ilnet import ILNet_L
        return ILNet_L(out_ch=1), (lambda m, x: m(x)), 3
    if kind == 'rpcanetpp':
        from models import get_model
        return get_model('rpcanet_pp'), (lambda m, x: m(x)[1]), 1
    if kind == 'l2sknet':
        from net import Net
        return Net('L2SKNet_UNet'), (lambda m, x: m(x)), 1
    if kind == 'bgm':
        from net import Net
        return Net(model_name='BasicUNet_plus'), (lambda m, x: m(x)), 1
    if kind == 'drpcanet':
        from models import get_model
        return get_model('Drpcanet'), (lambda m, x: m(x)[1]), 1
    if kind == 'durd':
        from utils.DURD import DURD
        return DURD(in_channel=1), (lambda m, x: m(x)[6]), 1
    raise SystemExit(f'unknown kind {kind}')


class Wrapped(nn.Module):
    """Single-output view of a baseline, so thop/FlopCounter can profile it."""

    def __init__(self, net, fwd):
        super().__init__()
        self.net = net
        self._fwd = fwd

    def forward(self, x):
        return self._fwd(self.net, x)


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


def flops_thop(model, x):
    from thop import profile
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return float(macs)


def flops_torch(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x)
    return float(counter.get_total_flops())


def time_round(model, x, warmup, runs):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def clock_warm(device, seconds=2.0):
    """Sustained matmul load so the GPU leaves its idle clock (180 MHz) state.

    Single-image microbenchmarks are too short to trigger boost; without this the
    measured latency is 2-4x worse than steady state on an idle card.
    """
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.mm(a, b)
    torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--clock-warm-s', type=float, default=2.0)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    net, fwd, ch = build(args.kind)
    net = net.to(device).eval()
    model = Wrapped(net, fwd).to(device).eval()
    x = torch.randn(1, ch, args.size, args.size, device=device)

    total, trainable = count_params(net)
    with torch.no_grad():
        fwd(net, x)

    try:
        macs = flops_thop(model, x)
        flops_src = 'thop(MACs)'
    except Exception as exc:  # noqa: BLE001
        macs = flops_torch(model, x) / 2.0
        flops_src = f'thop failed ({type(exc).__name__}); torch FlopCounter/2'

    if args.clock_warm_s > 0:
        clock_warm(device, args.clock_warm_s)

    best = None
    all_medians = []
    for _ in range(args.rounds):
        clock_warm(device, 0.4)
        times = time_round(model, x, args.warmup, args.runs)
        med = statistics.median(times)
        all_medians.append(med)
        if best is None or med < best['median']:
            best = {'mean': statistics.mean(times), 'median': med, 'min': min(times)}

    rec = {
        'kind': args.kind,
        'input': f'{ch}x{args.size}x{args.size}',
        'params_M': round(total / 1e6, 3),
        'gflops': round(macs / 1e9, 3),
        'gflops_source': flops_src,
        'latency_ms_median': round(best['median'], 3),
        'latency_ms_mean': round(best['mean'], 3),
        'latency_ms_min': round(best['min'], 3),
        'fps': round(1000.0 / best['median'], 2),
        'fps_mean_based': round(1000.0 / best['mean'], 2),
        'round_medians_ms': [round(m, 3) for m in all_medians],
        'runs_per_round': args.runs,
        'clock_warm_s': args.clock_warm_s,
        'device': torch.cuda.get_device_name(device),
    }
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == '__main__':
    main()
