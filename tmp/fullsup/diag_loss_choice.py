import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/share_datasets/Seal_Model_Reproduction/ACM-main')
from model.segmentation import ASKCResNetFPN
from model.loss import SoftLoULoss
from seal_dataset import SealIRSTDDataset
from train_seal import load_image
import os.path as osp

tr = SealIRSTDDataset('train')
full = load_image(osp.join(tr.imgs_dir, '1050.png'))
gt = np.asarray(tr.raw(tr.names.index('1050'))[2])
ys, xs = np.where(gt > 127)
cy, cx = int(ys.mean()), int(xs.mean())
y0 = max(0, min(417 - 256, cy - 128)); x0 = max(0, min(622 - 256, cx - 128))
patch = full[:, y0:y0 + 256, x0:x0 + 256]
lab = torch.from_numpy((gt[y0:y0 + 256, x0:x0 + 256] > 127).astype(np.float32))[None]
print('patch target px', int(lab.sum()))

def run(name, use_bce, steps=200):
    net = ASKCResNetFPN([4,4,4],[8,16,32,64],'AsymBi')
    net.apply(lambda m: (nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, nn.Conv2d) else
                         (nn.init.normal_(m.weight, 1.0, 0.02) if isinstance(m, nn.BatchNorm2d) else
                          (nn.init.normal_(m.bias, 0) if isinstance(m, nn.BatchNorm2d) else None))))
    net = net.cuda().train()
    data = patch.unsqueeze(0).repeat(8, 1, 1, 1).cuda()
    labels = lab.unsqueeze(0).repeat(8, 1, 1, 1).cuda()
    opt = torch.optim.Adagrad(net.parameters(), lr=0.05, weight_decay=1e-4)
    siou = SoftLoULoss(); bce = nn.BCEWithLogitsLoss()
    for step in range(1, steps + 1):
        out = net(data)
        loss = siou(out, labels)
        if use_bce:
            loss = loss + bce(out, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                p = torch.sigmoid(out) > 0.5
                g = labels > 0.5
                iou = float((p & g).sum()) / max(float((p | g).sum()), 1)
            print('%s step %d loss %.4f iou %.4f predfrac %.5f' % (name, step, loss.item(), iou, float(p.float().mean())))

run('official', False)
run('composite', True)
