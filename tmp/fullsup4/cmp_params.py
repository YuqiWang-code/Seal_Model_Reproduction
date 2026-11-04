import os
import sys

import torch
from thop import profile

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/tools')
import measure_complexity_one as M  # noqa: E402

kind = sys.argv[1]
net, fwd, cin = M.build(kind)
net = net.eval()
total = sum(p.numel() for p in net.parameters())
buf = sum(b.numel() for b in net.buffers())
x = torch.randn(1, cin, 256, 256)
wrapped = M.Wrapped(net, fwd)
with torch.no_grad():
    macs, thop_params = profile(wrapped, inputs=(x,), verbose=False)
print(f'{kind}: total_params_M={total / 1e6:.6f} buffers_M={buf / 1e6:.6f} '
      f'thop_params_M={thop_params / 1e6:.6f} thop_macs_G={macs / 1e9:.6f}', flush=True)
