import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/tools')
import measure_complexity_one as M  # noqa: E402

torch.backends.cudnn.benchmark = True
a = torch.randn(4096, 4096, device='cuda:1')
b = torch.randn(4096, 4096, device='cuda:1')
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2:
    torch.mm(a, b)
torch.cuda.synchronize()

kind = sys.argv[1]
net, fwd, _ = M.build(kind)
net = net.to('cuda:1').eval()
for bs in (1, 2, 4):
    x = torch.randn(bs, 1, 256, 256, device='cuda:1')
    with torch.no_grad():
        for _ in range(10):
            fwd(net, x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            torch.cuda.synchronize()
            s = time.perf_counter()
            fwd(net, x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    print(kind, 'bs', bs, 'median_ms', round(ts[len(ts) // 2], 2), 'min_ms', round(ts[0], 2), flush=True)
