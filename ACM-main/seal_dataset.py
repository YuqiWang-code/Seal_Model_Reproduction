"""Seal (thermal) dataset adapter for ACM.

Reads the shared IRSTD copy of Seal:

    /share_datasets/Seal_Model_Reproduction/Seal_IRSTD/
        images/<stem>.png  masks/<stem>.png  trainval.txt val.txt test.txt

Training keeps the official acm-pytorch augmentation (random mirror, random
scale on the long edge, random crop, Gaussian blur). Validation and test use the
native-resolution image: the evaluator slides a 256x256 window instead of
resizing, because Seal targets span only a few pixels.
"""
import os.path as osp
import random

import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

SEAL_ROOT = '/share_datasets/Seal_Model_Reproduction/Seal_IRSTD'


class SealIRSTDDataset(Data.Dataset):
    def __init__(self, mode='train', crop_size=256, base_size=256, root=SEAL_ROOT):
        assert mode in ('train', 'val', 'test')
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        self.base_size = base_size
        list_file = {'train': 'trainval.txt', 'val': 'val.txt', 'test': 'test.txt'}[mode]
        self.names = [line.strip() for line in open(osp.join(root, list_file)) if line.strip()]
        self.imgs_dir = osp.join(root, 'images')
        self.labels_dir = osp.join(root, 'masks')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        """Native-resolution RGB image and binary mask (PIL)."""
        name = self.names[index]
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('RGB')
        mask = Image.open(osp.join(self.labels_dir, name + '.png')).convert('L')
        return name, img, mask

    def __getitem__(self, index):
        name, img, mask = self.raw(index)
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
            img, mask = self.transform(img), self.to_tensor(mask)
        else:
            img, mask = np.asarray(img), np.asarray(mask)
            img, mask = self.transform(img), self.to_tensor(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask
