"""Fully-supervised ILNet training on the Seal thermal band.

Protocol (see docs/ILNet_Seal实验记录.md):
  * training data: Seal train split (1,500 images, full pixel GT)
  * model selection: Seal val split (500 images), pixel IoU at 0.5 with the same
    sliding-window inference used for test
  * schedule: official ILNet-L settings (Adan 1e-3 with warm-up cosine, BCE on the
    side outputs, batch 8), 300 epochs instead of the official 600, 256 crops
    instead of the 512 resize
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image

from model.ilnet import ILNet_L
from model.loss import criterion
from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits
from utils.Adan import Adan
from utils.lr_scheduler import create_lr_scheduler

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--val-interval', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', default='experiments/Seal_fullsup')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def load_image(path):
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1))


@torch.no_grad()
def val_iou(net, dataset, device, window, stride):
    net.eval()
    inter = union = 0.0
    for index in range(len(dataset)):
        _, _, mask = dataset.raw(index)
        tensor = load_image(osp.join(dataset.imgs_dir, dataset.names[index] + '.png'))
        logits = sliding_window_logits(net, tensor, window=window, stride=stride, device=device)
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

    net = ILNet_L(out_ch=1).to(device)
    optimizer = Adan(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True,
                                    warmup_epochs=5)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            preds = net(data)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses):.4f}', flush=True)

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
