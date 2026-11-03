"""Seal (thermal) dataset adapter for ILNet.

ILNet's official loader uses SIRST-specific mean/std; here the shared ImageNet
statistics of the other fully-supervised baselines are used so that every method
sees the same input, and 256x256 crops replace the official 512x512 resize.
"""
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        name = self.names[index]
        img = Image.open(osp.join(self.imgs_dir, name + '.png')).convert('RGB')
        mask = Image.open(osp.join(self.labels_dir, name + '.png')).convert('L')
        return name, img, mask

    def __getitem__(self, index):
        name, img, mask = self.raw(index)
        if self.mode == 'train':
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            c = self.crop_size
            if w < c or h < c:
                pad = Image.new('RGB', (max(c, w), max(c, h)), 0)
                pad.paste(img, (0, 0))
                img = pad
                padm = Image.new('L', (max(c, w), max(c, h)), 0)
                padm.paste(mask, (0, 0))
                mask = padm
                w, h = img.size
            x1 = random.randint(0, w - c)
            y1 = random.randint(0, h - c)
            img = img.crop((x1, y1, x1 + c, y1 + c))
            mask = mask.crop((x1, y1, x1 + c, y1 + c))
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return self.transform(img), self.to_tensor(mask)
