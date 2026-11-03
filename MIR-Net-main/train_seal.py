"""Fine-tune MIR-Net with Seal box labels and select by validation Fw."""

import argparse
import sys
import json
import shutil
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import transform
from net import Net
from train_processes import train_loss


class SealDataset(Dataset):
    def __init__(self, data_root, split, labels=None, size=384, dense_labels=False):
        self.image_dir = data_root / split / 'RGB'
        self.mask_dir = labels if split == 'train' else data_root / split / 'GT'
        self.images = sorted(self.image_dir.glob('*.jpg'))
        self.train = split == 'train'
        self.dense_labels = dense_labels
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 256
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 256
        operations = [transform.Normalize(self.mean, self.std),
                      transform.Resize(size, size)]
        if self.train:
            operations.append(transform.RandomHorizontalFlip())
        operations.append(transform.ToTensor())
        self.convert = transform.Compose(*operations)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / f'{image_path.stem}.png'
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        image, label = self.convert(image, np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32))
        if self.train:
            if self.dense_labels:
                label = (label > 127).float()
            else:
                label[label == 0] = 255
                label[label == 2] = 0
            return image, label
        return image, mask, mask_path.name, (height, width)


def validate(model, loader, output, evaluate_module):
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for image, mask, names, shapes in loader:
            height, width = int(shapes[0][0]), int(shapes[1][0])
            fg, bg = model(image.cuda(), shape=(height, width))
            prediction = torch.sigmoid(fg[0] - bg[0]).squeeze().cpu().numpy()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            cv2.imwrite(str(output / names[0]), (prediction * 255).astype(np.uint8))
    model.train()
    return evaluate_module.weighted_f_measure_folder(str(output), str(loader.dataset.mask_dir))['mean_Fw']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--initial_checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay_epoch', type=int, default=0)
    parser.add_argument('--dense_labels', action='store_true')
    parser.add_argument('--ce_input', choices=['logits', 'probabilities'], default='logits')
    args = parser.parse_args()
    args.mode = 'train'
    args.pretrained_backbone = None
    args.output.mkdir(parents=True, exist_ok=True)

    train = DataLoader(SealDataset(args.data_root, 'train', args.labels, args.trainsize,
                                   dense_labels=args.dense_labels),
                       batch_size=args.batchsize, shuffle=True, num_workers=4)
    val = DataLoader(SealDataset(args.data_root, 'val', size=args.trainsize),
                     batch_size=1, shuffle=False, num_workers=4)
    model = Net(args).cuda()
    model.load_state_dict(torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    evaluator_path = args.output / 'evaluate_source.py'
    shutil.copyfile(args.data_root.parent / 'evaluate.py', evaluator_path)
    spec = importlib.util.spec_from_file_location('fixed_evaluate', evaluator_path)
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    configuration.update(evaluator=str(evaluator_path), train_count=len(train.dataset),
                         val_count=len(val.dataset), optimizer='AdamW', weight_decay=1e-4)
    (args.output / 'config.json').write_text(json.dumps(configuration, indent=2))
    for filename in ['train_seal.py', 'train_processes.py', 'infer_seal.py', 'net.py']:
        shutil.copyfile(Path(__file__).parent / filename, args.output / filename)

    best_fw = -1.0
    for epoch in range(1, args.epoch + 1):
        if args.lr_decay_epoch:
            for group in optimizer.param_groups:
                group['lr'] = args.lr * (0.1 ** (epoch // args.lr_decay_epoch))
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for index, (image, label) in enumerate(train, 1):
            losses = train_loss(image.cuda().float(), label.cuda(), model, ce_input=args.ce_input)
            loss = losses[0] + 0.8 * losses[1] + 0.6 * losses[2] + 0.4 * losses[3]
            (loss / args.accumulation).backward()
            total_loss += loss.item()
            if index % args.accumulation == 0 or index == len(train):
                optimizer.step()
                optimizer.zero_grad()
        print(f'epoch {epoch}/{args.epoch} loss={total_loss / len(train):.5f}', flush=True)
        if epoch % 5 == 0 or epoch == args.epoch:
            checkpoint = args.output / f'model-epoch-{epoch:03d}.pth'
            torch.save(model.state_dict(), checkpoint)
            fw = validate(model, val, args.output / 'val_maps' / f'epoch-{epoch:03d}', evaluate)
            record = dict(epoch=epoch, checkpoint=str(checkpoint), val_Fw=float(fw),
                          lr=optimizer.param_groups[0]['lr'])
            with (args.output / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(f'epoch {epoch} val Fw={fw:.5f}', flush=True)
            if fw > best_fw:
                best_fw = fw
                shutil.copyfile(checkpoint, args.output / 'model-best.pth')
                (args.output / 'selection.json').write_text(json.dumps(record, indent=2))
                print(f'best epoch {epoch} Fw={best_fw:.5f}', flush=True)


if __name__ == '__main__':
    main()
