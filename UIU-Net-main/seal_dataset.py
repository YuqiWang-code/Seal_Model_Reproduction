"""Seal (thermal) dataset adapter for UIU-Net.

UIU-Net's official loader normalises every sample by its own maximum and then
applies the ImageNet mean/std of the first channel. Training crops 256x256 from
the native image with random mirroring; validation and test use the full frame
and a sliding window, because Seal targets cover only a few pixels.
"""
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image

SEAL_ROOT = '/share_datasets/Seal_Model_Reproduction/Seal_IRSTD'
MEAN_RGB = (0.485, 0.456, 0.406)
STD_RGB = (0.229, 0.224, 0.225)


def normalize(img_uint8, per_sample_max=True):
    """Replicates ToTensorLab(flag=0): divide by max, then ImageNet normalise."""
    x = img_uint8.astype(np.float32)
    if per_sample_max:
        peak = x.max()
        x = x / (peak if peak > 1e-6 else 1.0)
    if x.ndim == 2:
        x = x[..., None]
    chw = np.stack([(x[..., i] - mean) / std
                    for i, (mean, std) in enumerate(zip(MEAN_RGB, STD_RGB))], axis=0)
    return torch.from_numpy(chw.astype(np.float32))


class SealIRSTDDataset(Data.Dataset):
    def __init__(self, mode='train', crop_size=256, root=SEAL_ROOT):
        assert mode in ('train', 'val', 'test')
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        list_file = {'train': 'trainval.txt', 'val': 'val.txt', 'test': 'test.txt'}[mode]
        self.names = [line.strip() for line in open(osp.join(root, list_file)) if line.strip()]
        self.imgs_dir = osp.join(root, 'images')
        self.labels_dir = osp.join(root, 'masks')

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        name = self.names[index]
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('RGB')
        mask = Image.open(osp.join(self.labels_dir, name + '.png')).convert('L')
        return name, img, mask

    def __getitem__(self, index):
        name, img, mask = self.raw(index)
        img, mask = np.asarray(img), np.asarray(mask)
        if self.mode == 'train':
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            h, w = mask.shape
            c = self.crop_size
            if h < c or w < c:
                img = np.pad(img, ((0, max(0, c - h)), (0, max(0, c - w)), (0, 0)))
                mask = np.pad(mask, ((0, max(0, c - h)), (0, max(0, c - w))))
                h, w = mask.shape
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            img = img[top:top + c, left:left + c]
            mask = mask[top:top + c, left:left + c]
        return normalize(img), torch.from_numpy((mask > 127).astype(np.float32)[None])

    def load_full(self, index):
        name = self.names[index]
        img = np.asarray(Image.open(osp.join(self.imgs_dir, name + '.png')).convert('RGB'))
        return name, normalize(img, per_sample_max=True)
