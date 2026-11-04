import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/tools')
import measure_complexity_one as M  # noqa: E402

kind = sys.argv[1]
device = 'cuda:1'
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2:
    torch.mm(a, b)
torch.cuda.synchronize()

net, fwd, cin = M.build(kind)
net = net.to(device).eval()
x = torch.randn(1, cin, 256, 256, device=device)
with torch.no_grad():
    for _ in range(30):
        fwd(net, x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(100):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fwd(net, x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - s) * 1000)
ts.sort()
lo = sum(1 for t in ts if t < (ts[0] + ts[-1]) / 2)
print(f'{kind}: min={ts[0]:.2f} p10={ts[10]:.2f} p50={ts[50]:.2f} p90={ts[90]:.2f} max={ts[-1]:.2f} '
      f'fast_share={lo}/100', flush=True)
print('  first 20:', [round(t, 1) for t in ts[:20]], flush=True)
