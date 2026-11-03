"""Fully-supervised UIU-Net training on the Seal thermal band.

Protocol (see docs/UIU-Net_Seal实验记录.md):
  * training data: Seal train split (1,500 images, full pixel GT)
  * model selection: Seal val split (500 images), pixel IoU at 0.5 with the same
    sliding-window inference used for test; the test split is never read here
  * schedule: official UIU-Net settings (Adam 1e-3, BCE over the seven deep
    outputs, 500 epochs), crop 256 (the official loader resizes to 320 and crops
    288; Seal frames are 417x622 so native-scale 256 crops are used),
    batch 8 instead of 3
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from model import UIUNET
from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', default='experiments/Seal_fullsup')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def val_iou(net, dataset, device, window, stride):
    net.eval()
    inter = union = 0.0
    for index in range(len(dataset)):
        name, _, mask = dataset.raw(index)
        _, tensor = dataset.load_full(index)
        logits = sliding_window_logits(
            net, tensor, window=window, stride=stride, device=device,
            output_fn=lambda out: torch.logit(out[0].clamp(1e-6, 1 - 1e-6)))
        pred = logits > 0.0
        gt = np.asarray(mask) > 127
        inter += np.logical_and(pred, gt).sum()
        union += np.logical_or(pred, gt).sum()
    net.train()
    return float(inter / max(union, 1.0))


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    with open(osp.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainset = SealIRSTDDataset('train', args.crop_size)
    valset = SealIRSTDDataset('val', args.crop_size)
    train_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f'train {len(trainset)} val {len(valset)}', flush=True)

    net = UIUNET(3, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = sum(criterion(out, labels) for out in outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)

        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'time': time.strftime('%H:%M:%S')}
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            iou = val_iou(net, valset, device, args.window, args.stride)
            record['val_iou'] = iou
            if iou > best:
                best = iou
                torch.save({'net': net.state_dict(), 'epoch': epoch, 'val_iou': iou},
                           osp.join(args.out, 'best.pth'))
            print(f'epoch {epoch} loss {np.mean(losses):.4f} val_iou {iou:.5f} best {best:.5f}', flush=True)
        else:
            print(f'epoch {epoch} loss {np.mean(losses):.4f}', flush=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    torch.save({'net': net.state_dict(), 'epoch': args.epochs - 1, 'val_iou': best},
               osp.join(args.out, 'last.pth'))
    print(f'best val IoU {best:.5f}')


if __name__ == '__main__':
    main()
