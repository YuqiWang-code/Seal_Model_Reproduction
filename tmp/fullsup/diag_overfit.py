import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/ACM-main')
from model.segmentation import ASKCResNetFPN
from model.loss import SoftLoULoss
from seal_dataset import SealIRSTDDataset
from train_seal import load_image
import os.path as osp

tr = SealIRSTDDataset('train')
# pick an image with a decent number of target pixels
best = None
for i in range(len(tr)):
    _, _, m = tr.raw(i)
    n = int((np.asarray(m) > 127).sum())
    if best is None or n > best[1]:
        best = (i, n)
    if i > 200:
        break
print('chosen', tr.names[best[0]], 'gt px', best[1])

full = load_image(osp.join(tr.imgs_dir, tr.names[best[0]] + '.png'))
gt = np.asarray(tr.raw(best[0])[2])
ys, xs = np.where(gt > 127)
cy, cx = int(ys.mean()), int(xs.mean())
print('centroid', cy, cx)
y0 = max(0, min(417 - 256, cy - 128)); x0 = max(0, min(622 - 256, cx - 128))
patch = full[:, y0:y0 + 256, x0:x0 + 256]
lab = torch.from_numpy((gt[y0:y0 + 256, x0:x0 + 256] > 127).astype(np.float32))[None]
print('patch target px', int(lab.sum()))

net = ASKCResNetFPN([4,4,4],[8,16,32,64],'AsymBi')
net.apply(lambda m: (nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, nn.Conv2d) else
                     (nn.init.normal_(m.weight, 1.0, 0.02) if isinstance(m, nn.BatchNorm2d) else
                      (nn.init.normal_(m.bias, 0) if isinstance(m, nn.BatchNorm2d) else None))))
net = net.cuda().train()
data = patch.unsqueeze(0).repeat(8, 1, 1, 1).cuda()
labels = lab.unsqueeze(0).repeat(8, 1, 1, 1).cuda()
opt = torch.optim.Adagrad(net.parameters(), lr=0.05, weight_decay=1e-4)
siou = SoftLoULoss(); bce = nn.BCEWithLogitsLoss()
for step in range(1, 301):
    out = net(data)
    loss = siou(out, labels) + bce(out, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0:
        with torch.no_grad():
            p = torch.sigmoid(out) > 0.5
            g = labels > 0.5
            iou = float((p & g).sum()) / max(float((p | g).sum()), 1)
        print('step %d loss %.4f bce %.4f iou %.4f predfrac %.5f' % (
            step, loss.item(), bce(out, labels).item(), iou, float(p.float().mean())))
