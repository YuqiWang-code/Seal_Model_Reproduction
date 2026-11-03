"""Fully-supervised MSHNet training on the Seal thermal band.

Protocol (see docs/MSHNet_Seal实验记录.md):
  * training data: Seal train split (1,500 images, full pixel GT)
  * model selection: Seal val split (500 images), pixel IoU at 0.5 with the same
    sliding-window inference used for test; the test split is never read here
  * schedule: official MSHNet settings (400 epochs, Adagrad 0.05, warm-up 5
    epochs before the SLSIoU shape loss, multi-scale deep supervision),
    batch 8 instead of the paper's 4 to use the RTX 5090 memory
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

from model.MSHNet import MSHNet
from model.loss import SLSIoULoss
from seal_dataset import SealIRSTDDataset
from seal_infer import sliding_window_logits

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--warm-epoch', type=int, default=5)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--base-size', type=int, default=256)
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
        logits = sliding_window_logits(
            net, tensor, window=window, stride=stride, device=device,
            forward_fn=lambda m, x: m(x, True)[1])
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

    trainset = SealIRSTDDataset('train', args.crop_size, args.base_size)
    valset = SealIRSTDDataset('val', args.crop_size, args.base_size)
    train_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f'train {len(trainset)} val {len(valset)}', flush=True)

    net = MSHNet(3).to(device)
    criterion = SLSIoULoss()
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    down = torch.nn.MaxPool2d(2, 2)

    best = -1.0
    log_path = osp.join(args.out, 'validation.jsonl')
    open(log_path, 'w').close()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        warm_flag = epoch > args.warm_epoch
        for step, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)
            masks, pred = net(data, warm_flag)
            loss = criterion(pred, labels, args.warm_epoch, epoch)
            for j in range(len(masks)):
                if j > 0:
                    labels = down(labels)
                loss = loss + criterion(masks[j], labels, args.warm_epoch, epoch)
            loss = loss / (len(masks) + 1)
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
