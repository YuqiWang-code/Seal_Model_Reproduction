"""Fully-supervised DURD training on the Seal thermal band (TGRS 2026).

DURD unrolls a low-rank + sparse decomposition: the loss combines the sparse
prediction (SoftIoU) with the reconstruction term (MSE on L_out, weight 0.1), as
in the official `utils/tools.DURD_Loss`.
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.utils.data as Data

from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits
from utils.DURD import DURD
from utils.tools import SoftIoULoss

PRED_IDX = 6   # DURD.forward -> L_out, S_out, X_recon, S_prior, M, F_dec, final_pred
RECON_IDX = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=10)
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

    net = DURD(in_channel=1).to(device)
    # utils.tools.SoftIoULoss is a plain function (pred, target) in this release
    softiou = SoftIoULoss
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            outs = net(data)
            loss = softiou(outs[PRED_IDX], labels) + 0.1 * mse(outs[RECON_IDX], data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)
        scheduler.step()

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
                        output_fn=lambda o: o[PRED_IDX])
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
