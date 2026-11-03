import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from config import param as option
from dataset.dataloader import TestDataset
from model.model import get_model
from utils import set_seed


if __name__ == '__main__':
    set_seed(option)
    model = get_model(option)
    model.eval()
    output = Path(option['eval_save_path'])
    output.mkdir(parents=True, exist_ok=True)
    loader = TestDataset(option['paths']['test_root'], option['trainsize'],
                         with_gt=False, image_dir=option['test_image_dir'])
    with torch.no_grad():
        for _ in tqdm(range(loader.size), desc='Seal-test'):
            image, _, height, width, name = loader.load_data()
            prediction, _ = model(image.cuda(), (height, width))
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = 255 * (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            cv2.imwrite(str(output / name), prediction)
    print(f'Saved {loader.size} predictions to {output}')
