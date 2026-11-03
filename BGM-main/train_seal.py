"""Fully-supervised BGM training on the Seal thermal band (TGRS 2025).

BGM masks the background during training; its `Net` exposes the official loss via
`net.loss(pred, gt)`, which is what the loop uses here.
"""
import argparse

import torch

from net import Net
from seal_train_common import run

MODEL = 'BasicUNet_plus'


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
    net = Net(model_name=MODEL)
    criterion = lambda pred, gt: net.loss(pred, gt)  # noqa: E731
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 140], gamma=0.1)
    run(args, net, criterion, optimizer, scheduler, forward_fn=lambda m, x: m(x))


if __name__ == '__main__':
    main()
