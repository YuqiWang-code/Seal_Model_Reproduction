"""Fully-supervised DRPCA-Net training on the Seal thermal band (TGRS 2025).

Same unfolding family as RPCANet (sparse component + low-rank reconstruction).
Following the RPCANet observations on Seal, the region term is aggregated over the
whole batch and Adam uses 1e-4; everything else follows the official train.py.
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.utils.data as Data

from models import get_model
from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits


def batch_softiou(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum()
    return 1 - (inter + 1) / (pred.sum() + target.sum() - inter + 1 + eps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', default='experiments/Seal_fullsup')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


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

    net = get_model('Drpcanet').to(device)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    gamma = torch.tensor([0.01], device=device)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            out_D, out_T = net(data)
            loss = batch_softiou(out_T, labels) + gamma * mse(out_D, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (1 - (epoch + 1) / args.epochs) ** 0.9

        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'time': time.strftime('%H:%M:%S')}
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            net.eval()
            inter = union = 0.0
            with torch.no_grad():
                for index in range(len(valset)):
                    _, img, mask = valset.raw(index)
                    tensor = torch.from_numpy(np.ascontiguousarray(valset.normalize(np.asarray(img, np.float32))))[None]
                    logits = sliding_window_logits(
                        net, tensor, window=args.window, stride=args.stride, device=device,
                        output_fn=lambda o: o[1] if isinstance(o, tuple) else o)
                    pred = logits > 0.0
                    gt = np.asarray(mask) > 127
                    inter += np.logical_and(pred, gt).sum()
                    union += np.logical_or(pred, gt).sum()
            iou = float(inter / max(union, 1.0))
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
