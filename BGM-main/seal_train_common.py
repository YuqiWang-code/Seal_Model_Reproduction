"""Generic Seal training loop for the 2025/2026 fully-supervised baselines.

Each method folder keeps its own `train_seal.py` (model/loss/optimizer differ),
but the dataset, sliding-window evaluation and checkpoint bookkeeping are shared:

  * training data: Seal train split (1,500 images, full pixel GT)
  * selection: Seal val split (500 images), pixel IoU@0.5 with the same sliding
    window used at test time; the test split is never read here

This module is copied into the L2SKNet / BGM / DURD / DRPCA-Net folders.
"""
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.utils.data as Data

from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits


def batch_softiou(logits, target, eps=1e-6):
    """Dataset-level SoftIoU (DNA-Net style) for sparse foreground."""
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum()
    return 1 - (inter + 1) / (pred.sum() + target.sum() - inter + 1 + eps)


def run(args, net, criterion, optimizer, scheduler=None, forward_fn=None, logits_fn=None):
    """Generic train/validate loop.

    forward_fn(net, x) -> logits tensor used for the loss
    logits_fn(net, x)  -> logits tensor used for sliding-window inference
    """
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

    net = net.to(device)
    forward_fn = forward_fn or (lambda m, x: m(x))
    logits_fn = logits_fn or forward_fn

    def val_iou():
        net.eval()
        inter = union = 0.0
        with torch.no_grad():
            for index in range(len(valset)):
                _, img, mask = valset.raw(index)
                tensor = torch.from_numpy(np.ascontiguousarray(valset.normalize(np.asarray(img, np.float32))))[None]
                logits = sliding_window_logits(net, tensor, window=args.window, stride=args.stride,
                                               device=device, forward_fn=logits_fn)
                pred = logits > 0.0
                gt = np.asarray(mask) > 127
                inter += np.logical_and(pred, gt).sum()
                union += np.logical_or(pred, gt).sum()
        net.train()
        return float(inter / max(union, 1.0))

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            logits = forward_fn(net, data)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)
        if scheduler is not None:
            scheduler.step()

        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'time': time.strftime('%H:%M:%S')}
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            iou = val_iou()
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
