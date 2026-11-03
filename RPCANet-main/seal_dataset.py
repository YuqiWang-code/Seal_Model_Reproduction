"""Seal (thermal, single channel) dataset adapter for RPCANet / RPCANet++.

The official loader feeds one-channel images normalised to [0,1]. Training crops
256x256 from the native frame with random mirroring; validation and test keep the
native resolution and are evaluated with a sliding window.
"""
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image

SEAL_ROOT = '/share_datasets/Seal_Model_Reproduction/Seal_IRSTD'


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
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('L')
        mask = Image.open(osp.join(self.labels_dir, name + '.png')).convert('L')
        return name, img, mask

    def __getitem__(self, index):
        name, img, mask = self.raw(index)
        img, mask = np.asarray(img, np.float32), np.asarray(mask, np.float32)
        if self.mode == 'train':
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            h, w = mask.shape
            c = self.crop_size
            if h < c or w < c:
                img = np.pad(img, ((max(0, c - h), 0), (max(0, c - w), 0)))
                mask = np.pad(mask, ((max(0, c - h), 0), (max(0, c - w), 0)))
                h, w = mask.shape
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            img, mask = img[top:top + c, left:left + c], mask[top:top + c, left:left + c]
        return (torch.from_numpy(img / 255.0)[None],
                torch.from_numpy((mask > 127).astype(np.float32))[None])
