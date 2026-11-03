"""Seal (thermal, single channel) dataset adapter for SCTransNet.

Official preprocessing normalises the 0-255 image with the dataset mean/std
(`Normalized` in utils.py). The statistics below are measured on the 1,500 Seal
train images of Seal_IRSTD and cached in norm_stats.json.
"""
import json
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image

SEAL_ROOT = '/share_datasets/Seal_Model_Reproduction/Seal_IRSTD'
_STATS_CACHE = osp.join(osp.dirname(osp.abspath(__file__)), 'norm_stats.json')


def _compute_stats(root, names, sample=300):
    vals = []
    for name in names[:sample]:
        img = np.asarray(Image.open(osp.join(root, 'images', name + '.png')).convert('L'), np.float32)
        vals.append(img.ravel())
    allv = np.concatenate(vals)
    return {'mean': float(allv.mean()), 'std': float(allv.std())}


def get_norm_stats(root=SEAL_ROOT):
    if osp.exists(_STATS_CACHE):
        with open(_STATS_CACHE) as f:
            return json.load(f)
    names = [line.strip() for line in open(osp.join(root, 'trainval.txt')) if line.strip()]
    stats = _compute_stats(root, names)
    with open(_STATS_CACHE, 'w') as f:
        json.dump(stats, f)
    print('norm stats', stats, flush=True)
    return stats


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
        self.stats = get_norm_stats(root)

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        name = self.names[index]
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('L')
        mask = Image.open(osp.join(self.labels_dir, name + '.png')).convert('L')
        return name, img, mask

    def normalize(self, img_uint8):
        return (img_uint8.astype(np.float32) - self.stats['mean']) / self.stats['std']

    def __getitem__(self, index):
        name, img, mask = self.raw(index)
        img, mask = np.asarray(img, np.float32), np.asarray(mask, np.float32)
        img = self.normalize(img)
        if self.mode == 'train':
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            h, w = mask.shape
            c = self.crop_size
            if h < c or w < c:
                img = np.pad(img, ((0, max(0, c - h)), (0, max(0, c - w))))
                mask = np.pad(mask, ((0, max(0, c - h)), (0, max(0, c - w))))
                h, w = mask.shape
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            img, mask = img[top:top + c, left:left + c], mask[top:top + c, left:left + c]
        return (torch.from_numpy(np.ascontiguousarray(img))[None],
                torch.from_numpy((mask > 127).astype(np.float32))[None])
