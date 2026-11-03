import cv2
import torch
import numpy as np


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb, t, d):
        for op in self.ops:
            rgb, t, d = op(rgb, t, d)
        return rgb, t, d


class Normalize(object):
    def __init__(self, mean1, mean2, mean3, std1, std2, std3):
        self.mean1 = mean1
        self.mean2 = mean2
        self.mean3 = mean3
        self.std1 = std1
        self.std2 = std2
        self.std3 = std3

    def __call__(self, rgb, t, d):
        rgb = (rgb - self.mean1) / self.std1
        t = (t - self.mean2) / self.std2
        d = (d - self.mean3) / self.std3
        return rgb, t, d


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, t, d):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        t = cv2.resize(t, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        d = cv2.resize(d, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb, t, d


class ToTensor(object):
    def __call__(self, rgb, t, d):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)
        t = torch.from_numpy(t)
        t = t.permute(2, 0, 1)
        d = torch.from_numpy(d)
        d = d.permute(2, 0, 1)
        return rgb, t, d


# class Compose(object):
#     def __init__(self, *ops):
#         self.ops = ops
#
#     def __call__(self, rgb,t,d, mask):
#         for op in self.ops:
#             rgb,t,d, mask = op(rgb,t,d, mask)
#         return rgb,t,d, mask
#
# class Normalize(object):
#     def __init__(self, mean1,mean2,mean3, std1,std2,std3):
#         self.mean1 = mean1
#         self.mean2 = mean2
#         self.mean3 = mean3
#         self.std1 = std1
#         self.std2 = std2
#         self.std3 = std3
#
#     def __call__(self, rgb,t,d, mask):
#         rgb = (rgb - self.mean1)/self.std1
#         t = (t - self.mean2) / self.std2
#         d = (d - self.mean3) / self.std3
#         mask /= 255
#         return rgb,t,d, mask

class Minusmean(object):
    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, rgb, t, d, mask):
        rgb = rgb - self.mean1
        t = t - self.mean2
        d = d - self.mean1
        mask /= 255
        return rgb, t, d, mask


# class Resize(object):
#     def __init__(self, H, W):
#         self.H = H
#         self.W = W
#
#     def __call__(self, rgb,t,d, mask):
#         rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         t = cv2.resize(t, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         d = cv2.resize(d, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         return rgb, t, d, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, t, d, mask):
        H, W, _ = rgb.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        rgb = rgb[ymin:ymin + self.H, xmin:xmin + self.W, :]
        t = t[ymin:ymin + self.H, xmin:xmin + self.W, :]
        d = d[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return rgb, t, d, mask


class Random_rotate(object):
    def __call__(self, rgb, t, d, mask):
        angle = np.random.randint(-25, 25)
        h, w, _ = rgb.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(rgb, M, (w, h)), cv2.warpAffine(t, M, (w, h)), cv2.warpAffine(d, M,
                                                                                            (w, h)), cv2.warpAffine(
            mask, M, (w, h))


class RandomHorizontalFlip(object):
    def __call__(self, rgb, t, d, mask):
        if np.random.randint(2) == 1:
            rgb = rgb[:, ::-1, :].copy()
            t = t[:, ::-1, :].copy()
            d = d[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        return rgb, t, d, mask

# class ToTensor(object):
#     def __call__(self, rgb,t,d, mask):
#         rgb = torch.from_numpy(rgb)
#         rgb = rgb.permute(2, 0, 1)
#         t = torch.from_numpy(t)
#         t = t.permute(2, 0, 1)
#         d = torch.from_numpy(d)
#         d = d.permute(2, 0, 1)
#         mask = torch.from_numpy(mask)
#         mask = mask.permute(2, 0, 1)
#         return rgb,t,d,mask.mean(dim=0, keepdim=True)
