import os
import cv2
import time
import torch
import numpy as np
from dataset.dataloader import TestDataset
from tqdm import tqdm
from config import param as option
from model.model import get_model
from utils import set_seed

if __name__ == '__main__':
    set_seed(option)
    test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    model = get_model(option)
    model.eval()

    for dataset in option['datasets']:
        result_save_path = option['eval_save_path'] + test_epoch_num + '_epoch/' + dataset + '/'
        os.makedirs(result_save_path, exist_ok=True)
        test_root = option['paths']['test_root'] + dataset + '/'
        test_loader = TestDataset(test_root, option['trainsize'])

        time_list = []
        for i in tqdm(range(test_loader.size), desc=dataset):
            image, _, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            torch.cuda.synchronize()
            start = time.time()
            res, _ = model.forward(image, (HH,WW),name)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - start)
            cv2.imwrite(result_save_path + name, res)

        print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))



