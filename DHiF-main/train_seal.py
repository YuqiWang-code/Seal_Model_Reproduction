"""Fully-supervised DHiF (MTU backbone) training on the Seal thermal band.

Protocol (see docs/DHiF_Seal实验记录.md):
  * training data: Seal train split (1,500 images, full pixel GT)
  * model selection: Seal val split (500 images), pixel IoU at 0.5 with the same
    sliding-window inference used for test
  * schedule: official DHiF settings (MTU backbone + DHiF convolution, Adam 5e-4,
    MultiStepLR halving every 50 epochs, SoftIoU loss, 256 patches), 200 epochs
    instead of the official 500
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.utils.data as Data

from net import Net
from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits


def batch_softiou(logits, target, eps=1e-6):
    """Dataset-level SoftIoU (DNA-Net style).

    The official per-sample SoftIoU averages each 256 patch separately; on Seal
    (0.02% foreground, several tiny targets per frame) that gradient collapses to
    the all-background solution, so the region term is aggregated over the whole
    batch instead. Network, optimizer and schedule are unchanged.
    """
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum()
    return 1 - (inter + 1) / (pred.sum() + target.sum() - inter + 1 + eps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--conv', default='DHiF')
    p.add_argument('--model', default='MTU', help="official backbone: MTU (default) or MSH")
    p.add_argument('--out', default='experiments/Seal_fullsup')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def normalize_tensor(dataset, arr):
    return torch.from_numpy(np.ascontiguousarray(dataset.normalize(arr)))[None]


@torch.no_grad()
def val_iou(net, dataset, device, window, stride):
    net.eval()
    inter = union = 0.0
    for index in range(len(dataset)):
        _, img, mask = dataset.raw(index)
        tensor = normalize_tensor(dataset, np.asarray(img, np.float32))
        logits = sliding_window_logits(net, tensor, window=window, stride=stride, device=device,
                                       forward_fn=lambda m, x: m(x, -1))
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

    net = Net(model_name=args.model, mode='train', conv=args.conv, save_path=args.out).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(50, args.epochs, 50)), gamma=0.5)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            pred = net(data, epoch)
            logits = pred[-1] if isinstance(pred, (list, tuple)) else pred
            loss = batch_softiou(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)
        scheduler.step()

        record = {'epoch': epoch, 'loss': float(np.mean(losses)),
                  'lr': optimizer.param_groups[0]['lr'], 'time': time.strftime('%H:%M:%S')}
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
