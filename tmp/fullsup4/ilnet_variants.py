import os
import sys

import torch
from thop import profile

sys.path.insert(0, os.getcwd())
from model.ilnet import ILNet_L, ILNet_M, ILNet_S  # noqa: E402


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


for name, ctor in (('ILNet_S', ILNet_S), ('ILNet_M', ILNet_M), ('ILNet_L', ILNet_L)):
    net = ctor(out_ch=1).eval()
    total = sum(p.numel() for p in net.parameters())
    for res in (256,):
        x = torch.randn(1, 3, res, res)
        with torch.no_grad():
            macs, thop_params = profile(Wrapped(net), inputs=(x,), verbose=False)
        print(f'{name} @{res}: total_params_M={total / 1e6:.6f} thop_params_M={thop_params / 1e6:.6f} '
              f'macs_G={macs / 1e9:.6f}', flush=True)
