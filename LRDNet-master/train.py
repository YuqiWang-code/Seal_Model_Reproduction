import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
from dataset.dataloader import get_loader
from config import param as option
from utils import set_seed
from model.model import get_model
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from dataset.dataloader import TestDataset
from train_processes import *
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import weighted_f_measure_folder

BASE_LR = 1e-5
MAX_LR = 1e-3


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1.,annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def validate(model, val_loader, dataset, pred_dir, gt_dir):
    model.eval()
    avg_mae = 0.0
    cnt = 0
    os.makedirs(pred_dir, exist_ok=True)
    with torch.no_grad():
        for _ in tqdm(range(val_loader.size), desc=dataset):
            image, mask, HH, WW, name = val_loader.load_data()
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _ = model(image)
            out = F.interpolate(out, size=(HH, WW), mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out[0, 0])
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            avg_mae += torch.abs(pred - mask[0]).mean().item()
            cv2.imwrite(os.path.join(pred_dir, name), (pred.cpu().numpy() * 255).astype(np.uint8))
            cnt += len(image)
    model.train()

    fw = weighted_f_measure_folder(pred_dir, gt_dir)['mean_Fw']
    return (avg_mae / cnt), fw


def train_one_epoch(epoch, model, optimizer, train_loader, option, sw):
    model.train()
    batch_idx = -1
    db_size = len(train_loader)
    start_from = 0
    global_step = start_from * db_size
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for pack in progress_bar:
        niter = (epoch - 1) * db_size + batch_idx + 1
        lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, option['epoch'] * db_size, niter, ratio=1.)
        optimizer.param_groups[0]['lr'] = 0.01 * lr  # for backbone
        optimizer.param_groups[1]['lr'] = lr
        batch_idx += 1
        global_step += 1

        images, gts = pack['image'].cuda(), pack['gt'].cuda()
        images = F.interpolate(images, size=option['trainsize'], mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, size=option['trainsize'], mode='bilinear', align_corners=True)
        loss2,loss3,loss4,loss5,loss6 = train_loss(image=images, mask=gts, net=model, ctx=dict(
            epoch=epoch, global_step=epoch, sw=sw, t_epo=option['epoch'], second_time=option['second_time']))

        loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4 + loss6 * 0.2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        sw.add_scalar('loss', loss.item(), global_step=epoch)
        progress_bar.set_postfix({'loss':f'{loss.item():.3e}', 'lr:':f"{optimizer.param_groups[0]['lr']:.2e}"})
    return model


if __name__ == "__main__":
    print('[INFO] Experiments saved in: ', option['log_path'])
    set_seed(option)
    train_loader = get_loader(option)
    model = get_model(option)

    base, head = [], []
    for name, param in model.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = AdamW([{'params': base, 'lr':option['lr_config']['lr']}, {'params': head}], option['lr_config']['lr'],
                      betas=option['lr_config']['beta'])

    os.makedirs(os.path.join(option['log_path'], 'tensorboard_log'), exist_ok=True)
    sw = SummaryWriter(os.path.join(option['log_path'], 'tensorboard_log'))

    best_fw = -1.0
    validation_path = os.path.join(option['log_path'], 'validation.csv')
    if option['validate_epoch']:
        with open(validation_path, 'w', newline='', encoding='utf-8') as file:
            csv.writer(file).writerow(['epoch', 'checkpoint', 'val_fw', 'val_mae'])
    for epoch in range(1, (option['epoch']+1)):

        model_weight = train_one_epoch(epoch, model, optimizer, train_loader, option, sw)

        # validate
        if option['validate_epoch'] and epoch % option['validate_epoch'] == 0:
            os.makedirs(option['ckpt_save_path'], exist_ok=True)
            checkpoint_path = os.path.join(option['ckpt_save_path'], f'{epoch:03d}_epoch.pth')
            torch.save(model_weight.state_dict(), checkpoint_path)
            test_loader = TestDataset(option['paths']['val_root'], option['trainsize'])
            mae, fw = validate(
                model, test_loader, 'Seal-val',
                os.path.join(option['log_path'], 'val_maps', f'epoch_{epoch:03d}'),
                os.path.join(option['paths']['val_root'], 'GT'))
            with open(validation_path, 'a', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow([epoch, checkpoint_path, f'{fw:.10f}', f'{mae:.10f}'])
            sw.add_scalar('Seal_validate_mae', mae, global_step=epoch)
            sw.add_scalar('Seal_validate_fw', fw, global_step=epoch)
            print('Seal_validate_mae: {} Seal_validate_fw: {}'.format(mae, fw))
            if fw > best_fw:
                best_fw = fw
                os.makedirs(option['ckpt_save_path'], exist_ok=True)
                torch.save(model.state_dict(), os.path.join(option['ckpt_save_path'], 'best.pth'))
                print('Seal_best_epoch: {}'.format(epoch))

        # save model.state_dict()
        save_path = option['ckpt_save_path']
        os.makedirs(save_path, exist_ok=True)
        if not option['validate_epoch'] and epoch >= (option['epoch'] - 30) and epoch % option['save_epoch'] == 0:
            save_name = os.path.join(save_path, '{:0>2d}_epoch.pth'.format(epoch))
            torch.save(model_weight.state_dict(), save_name)
    sw.close()
