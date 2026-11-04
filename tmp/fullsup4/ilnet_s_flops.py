import os
import sys

import torch
from thop import profile

sys.path.insert(0, os.getcwd())
from model.ilnet import ILNet_S  # noqa: E402


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


net = ILNet_S(out_ch=1).eval()
for res in (512, 256):
    x = torch.randn(1, 3, res, res)
    try:
        with torch.no_grad():
            macs, _ = profile(Wrapped(net), inputs=(x,), verbose=False)
        print(f'ILNet_S @{res}: macs_G={macs / 1e9:.4f}', flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f'ILNet_S @{res}: FAIL {type(exc).__name__}: {str(exc)[:120]}', flush=True)
