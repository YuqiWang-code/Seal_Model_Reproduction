"""Fully-supervised L2SKNet training on the Seal thermal band (TGRS 2025).

The official per-sample SoftIoU collapses to the all-background solution on Seal
(0.02% foreground): 20 epochs gave val IoU 0.0001 with the training loss still
decreasing. As already documented for the RPCANet family, the region term is
aggregated over the whole batch (DNA-Net style) and everything else follows the
official train_device0.py (Adam 5e-4, MultiStep [100,140], 256 patches).
"""
import argparse

import torch

from net import Net
from seal_train_common import batch_softiou, run

# official net.py maps the name below to the L2SKNet U-Net variant
MODEL = 'L2SKNet_UNet'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=10)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', default='experiments/Seal_fullsup')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    net = Net(MODEL)
    criterion = batch_softiou
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 140], gamma=0.1)
    run(args, net, criterion, optimizer, scheduler, forward_fn=lambda m, x: m(x))


if __name__ == '__main__':
    main()
