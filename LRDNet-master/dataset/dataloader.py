import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
from dataset.augment import randomFlip, randomCrop, randomRotation, colorEnhance, randomPeper


class TrainDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, second_time=False):
        self.trainsize = trainsize
        self.second_time = second_time
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.png'))])
        self.gts = [os.path.join(gt_root, os.path.splitext(os.path.basename(f))[0] + '.png')
                    for f in self.images]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Resize(
            (self.trainsize, self.trainsize),
            interpolation=transforms.InterpolationMode.BILINEAR if second_time
            else transforms.InterpolationMode.NEAREST)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')

        image, gt = randomFlip(image, gt)
        original_image, original_gt = image, gt
        image, gt = randomCrop(image, gt)
        if not self.second_time and not np.any(np.array(gt) == 1):
            image, gt = original_image, original_gt
        original_image, original_gt = image, gt
        image, gt = randomRotation(image, gt)
        if not self.second_time and not np.any(np.array(gt) == 1):
            image, gt = original_image, original_gt
        image = colorEnhance(image)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        gt = torch.from_numpy(np.array(gt, dtype=np.float32)).unsqueeze(0)
        if self.second_time:
            confidence = gt / 255.
            gt = torch.full_like(confidence, 255.)
            gt[confidence <= 0.15] = 0.
            gt[confidence >= 0.95] = 1.  # 0:bg, 1:fg, 255:unconfident
        else:
            gt[gt == 0.] = 255.
            gt[gt == 2.] = 0.  # 0:bg, 1:fg, 255:unknown
        return {'image': image, 'gt': gt}  # (3,320,320),(1,320,320)

    def __len__(self):
        return len(self.images)


def get_loader(option):
    dataset = TrainDataset(option['paths']['image_root'], option['paths']['gt_root'], trainsize=option['trainsize'],
                           second_time=option['second_time'])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['num_workers'],
                                  pin_memory=True)
    return data_loader


# out: B, C, H, W; data:[0-255]-> [0,1] ->[-1,1]
class TestDataset:
    def __init__(self, test_root, testsize, with_gt=True, image_dir='RGB'):
        self.testsize = testsize
        self.with_gt = with_gt
        image_root = os.path.join(test_root, image_dir)
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.png'))])
        if with_gt:
            gt_root = os.path.join(test_root, 'GT')
            self.gts = [os.path.join(gt_root, os.path.splitext(os.path.basename(f))[0] + '.png')
                        for f in self.images]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        gt = Image.open(self.gts[self.index]) if self.with_gt else None
        WW, HH = image.size[0], image.size[1]
        image = self.img_transform(image).unsqueeze(0)
        gt = self.gt_transform(gt) if self.with_gt else None
        name = os.path.splitext(os.path.basename(self.images[self.index]))[0] + '.png'
        self.index += 1
        return image, gt, HH, WW, name


class GenerateDataset:
    def __init__(self, generate_root, generate_size):
        self.generate_size = generate_size
        image_root = generate_root
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.png'))])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.generate_size, self.generate_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        WW, HH = image.size[0], image.size[1]
        image = self.img_transform(image).unsqueeze(0)
        name = os.path.basename(self.images[self.index])
        self.index += 1
        return image, HH, WW, name



class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        self.image_path = sorted([img_root + f for f in os.listdir(img_root)])
        self.label_path = sorted([label_root + f for f in os.listdir(label_root)])

    def __getitem__(self, item):
        pred_path = self.image_path[item]
        mask_path = self.label_path[item]
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (320,320)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
