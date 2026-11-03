"""Save RPCANet / RPCANet++ predictions for the Seal test split."""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import get_model
from seal_dataset import SealIRSTDDataset
from seal_infer import logits_to_uint8, sliding_window_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--net-name', default='rpcanet', choices=['rpcanet', 'rpcanet_pp'])
    p.add_argument('--weights', default='experiments/Seal_fullsup/best.pth')
    p.add_argument('--out_dir', default='/share_datasets/Seal_Model_Reproduction/results/RPCANet/Seal/fullsup_test_maps')
    p.add_argument('--split', default='test')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    net = get_model(args.net_name)
    state = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state['net'] if 'net' in state else state)
    net = net.to(device).eval()
    print(f"weights epoch={state.get('epoch')} val_iou={state.get('val_iou')}", flush=True)

    dataset = SealIRSTDDataset(args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    for index in tqdm(range(len(dataset))):
        _, img, _ = dataset.raw(index)
        tensor = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
        logits = sliding_window_logits(
            net, tensor, window=args.window, stride=args.stride, device=device,
            output_fn=lambda out: out[1] if isinstance(out, tuple) else out)
        cv2.imwrite(osp.join(args.out_dir, dataset.names[index] + '.png'), logits_to_uint8(logits))
    print(f'saved {len(dataset)} maps to {args.out_dir}')


if __name__ == '__main__':
    main()
