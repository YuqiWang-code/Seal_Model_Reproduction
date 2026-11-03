import cv2, numpy as np
from pathlib import Path
seal = Path('/share_datasets/Seal_Model_Reproduction/Seal_data')
root = Path('/share_datasets/Seal_Model_Reproduction/Seal_IRSTD')
names = ['1000', '1001', '1178', '1020', '1050']
for n in names:
    gt = cv2.imread(str(seal/'train'/'GT'/f'{n}.png'), 0) > 127
    t = cv2.imread(str(seal/'train'/'T'/f'{n}.tif'), cv2.IMREAD_UNCHANGED).astype(np.float32)
    rgb = cv2.imread(str(seal/'train'/'RGB'/f'{n}.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    t8 = cv2.imread(str(root/'images'/f'{n}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    def stats(x, tag):
        fg, bg = x[gt], x[~gt]
        print(f'  {tag}: fg mean {fg.mean():8.1f} std {fg.std():6.1f} | bg mean {bg.mean():8.1f} std {bg.std():6.1f} | '
              f'diff {fg.mean()-bg.mean():8.1f} | SNR {(fg.mean()-bg.mean())/bg.std():6.2f} | max {x.max():.0f}')
    print(n, 'gt px', gt.sum())
    stats(rgb, 'RGB ')
    stats(t, 'T16 ')
    stats(t8, 'T8  ')
