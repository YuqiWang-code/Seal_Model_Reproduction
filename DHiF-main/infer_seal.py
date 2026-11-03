"""Save DHiF predictions for the Seal test split (native resolution, 8-bit PNG)."""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

from net import Net
from seal_dataset import SealIRSTDDataset
from seal_infer import logits_to_uint8, sliding_window_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--conv', default='DHiF')
    p.add_argument('--weights', default='experiments/Seal_fullsup/best.pth')
    p.add_argument('--out_dir', default='/share_datasets/Seal_Model_Reproduction/results/DHiF/Seal/fullsup_test_maps')
    p.add_argument('--split', default='test')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    net = Net(model_name='MTU', mode='test', conv=args.conv)
    state = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state['net'] if 'net' in state else state)
    net = net.to(device).eval()
    print(f"weights epoch={state.get('epoch')} val_iou={state.get('val_iou')}", flush=True)

    dataset = SealIRSTDDataset(args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    for index in tqdm(range(len(dataset))):
        name, img, _ = dataset.raw(index)
        tensor = torch.from_numpy(np.ascontiguousarray(dataset.normalize(np.asarray(img, np.float32))))[None]
        logits = sliding_window_logits(net, tensor, window=args.window, stride=args.stride,
                                       device=device, forward_fn=lambda m, x: m(x, -1))
        cv2.imwrite(osp.join(args.out_dir, name + '.png'), logits_to_uint8(logits))
    print(f'saved {len(dataset)} maps to {args.out_dir}')


if __name__ == '__main__':
    main()
