"""Diagnose eval-mode vs train-mode val IoU for a finished checkpoint.

Usage: python diag_modes.py <method_dir> <weights> <net>
"""
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
method, weights, net_name = sys.argv[1], sys.argv[2], sys.argv[3]

from seal_dataset import SealIRSTDDataset  # noqa: E402
from seal_infer import sliding_window_logits  # noqa: E402

ds = SealIRSTDDataset('val')

if net_name == 'rpcanet':
    from models import get_model
    net = get_model('rpcanet')
    state = torch.load(weights, map_location='cpu')
    net.load_state_dict(state['net'] if 'net' in state else state)
    net = net.cuda().eval()
    out_fn = lambda o: o[1] if isinstance(o, tuple) else o  # noqa: E731
    get_tensor = lambda i: torch.from_numpy(np.asarray(ds.raw(i)[1], np.float32) / 255.0)[None]  # noqa: E731
elif net_name == 'sctransnet':
    from train_seal import build_model
    net = build_model('train')
    state = torch.load(weights, map_location='cpu')
    net.load_state_dict(state['net'] if 'net' in state else state)
    net = net.cuda().eval()
    out_fn = lambda o: torch.logit((o[-1] if isinstance(o, tuple) else o).clamp(1e-6, 1 - 1e-6))  # noqa: E731
    get_tensor = lambda i: torch.from_numpy(np.ascontiguousarray(ds.normalize(np.asarray(ds.raw(i)[1], np.float32))))[None]  # noqa: E731
else:
    raise SystemExit('unknown net')

print('epoch', state.get('epoch'), 'val_iou', state.get('val_iou'))
for mode in ('eval', 'train'):
    getattr(net, mode)()
    inter = union = 0.0
    frac = []
    for i in range(40):
        t = get_tensor(i)
        lg = sliding_window_logits(net, t, device='cuda:0', output_fn=out_fn)
        p = lg > 0
        gt = np.asarray(ds.raw(i)[2]) > 127
        inter += (p & gt).sum(); union += (p | gt).sum()
        frac.append(p.mean())
    print(f'{mode}: IoU {inter / max(union, 1):.5f} pred_frac {np.mean(frac):.6f}')
