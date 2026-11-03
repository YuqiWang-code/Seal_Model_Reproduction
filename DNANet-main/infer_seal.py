"""Save DNA-Net predictions for the Seal test split (native resolution, 8-bit PNG)."""
import argparse
import os
import os.path as osp

import cv2
import torch
from tqdm import tqdm

from model.model_DNANet import DNANet, Res_CBAM_block
from seal_dataset import SealIRSTDDataset
from seal_infer import logits_to_uint8, sliding_window_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--weights', default='experiments/Seal_fullsup/best.pth')
    p.add_argument('--out_dir', default='/share_datasets/Seal_Model_Reproduction/results/DNANet/Seal/fullsup_test_maps')
    p.add_argument('--split', default='test')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    net = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                 nb_filter=[16, 32, 64, 128, 256], deep_supervision='True')
    state = torch.load(args.weights, map_location='cpu')
    net.load_state_dict(state['net'] if 'net' in state else state)
    net = net.to(device).eval()
    print(f"weights epoch={state.get('epoch')} val_iou={state.get('val_iou')}", flush=True)

    dataset = SealIRSTDDataset(args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    for index in tqdm(range(len(dataset))):
        name = dataset.names[index]
        tensor, _ = dataset[index]
        logits = sliding_window_logits(
            net, tensor, window=args.window, stride=args.stride, device=device,
            output_fn=lambda out: out[-1] if isinstance(out, list) else out)
        cv2.imwrite(osp.join(args.out_dir, name + '.png'), logits_to_uint8(logits))
    print(f'saved {len(dataset)} maps to {args.out_dir}')


if __name__ == '__main__':
    main()
