#!/usr/bin/env python3
import cv2, numpy as np
from pathlib import Path
root = Path('/share_datasets/Seal_Model_Reproduction/Seal_IRSTD')
seal = Path('/share_datasets/Seal_Model_Reproduction/Seal_data')
names = ['1000', '1001', '1002', '1003']
rows = []
for n in names:
    img = cv2.imread(str(root/'images'/f'{n}.png'))
    gt = cv2.imread(str(root/'masks'/f'{n}.png'), 0)
    raw = cv2.imread(str(seal/'train'/'T'/f'{n}.tif'), cv2.IMREAD_UNCHANGED)
    raw8 = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    raw8 = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
    gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(gt > 0)
    for im in (img, raw8):
        if len(xs):
            y0, y1, x0, x1 = ys.min()-20, ys.max()+20, xs.min()-20, xs.max()+20
            cv2.rectangle(im, (x0, min(y0, 416)), (x1, max(y1,0)), (0, 0, 255), 1)
    rows.append(np.hstack([img, raw8, gt3]))
out = np.vstack(rows)
out = cv2.resize(out, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
cv2.imwrite('/share_datasets/Seal_Model_Reproduction/tmp_repos/preview.png', out)
print('ok', out.shape)
