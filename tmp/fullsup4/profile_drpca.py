import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/tools')
import measure_complexity_one as M  # noqa: E402

net, fwd, _ = M.build('drpcanet')
net = net.to('cuda:1').eval()
x = torch.randn(1, 1, 256, 256, device='cuda:1')
with torch.no_grad():
    for _ in range(10):
        fwd(net, x)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            fwd(net, x)
        torch.cuda.synchronize()

rows = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
total = sum(e.self_device_time_total for e in rows) / 20.0
print(f'total per forward: {total / 1000:.2f} ms')
for e in rows[:12]:
    if e.self_device_time_total <= 0:
        continue
    print(f'{e.key[:60]:60s} {e.self_device_time_total / 20 / 1000:8.3f} ms '
          f'{100 * e.self_device_time_total / (total * 20):5.1f}%  calls/iter={e.count // 20}')
