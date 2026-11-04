import os
import sys

sys.path.insert(0, os.getcwd())
from model.ilnet import ILNet_L, ILNet_M, ILNet_S  # noqa: E402

for name, ctor in (('ILNet_S', ILNet_S), ('ILNet_M', ILNet_M), ('ILNet_L', ILNet_L)):
    net = ctor(out_ch=1)
    total = sum(p.numel() for p in net.parameters())
    print(f'{name}: params_M={total / 1e6:.6f}', flush=True)
