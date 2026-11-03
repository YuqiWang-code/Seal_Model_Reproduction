"""How much IoU does a one-pixel boundary error cost on Seal GT?

IoU is computed with the project's whole-dataset pixel accumulation, but on a
synthetic prediction = 1-pixel dilation (or erosion) of the GT, so it isolates
the boundary sensitivity of targets whose mean area is ~14 px.
"""

from pathlib import Path

import cv2
import numpy as np

gt_dir = Path('Ours/Seal_test_GT')
kernel8 = np.ones((3, 3), np.uint8)

totals = {'gt': 0, 'dil': 0, 'ero': 0, 'inter_dil': 0, 'inter_ero': 0}
for path in sorted(gt_dir.glob('*.png')):
    gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 128
    gt_u8 = gt.astype(np.uint8)
    dil = cv2.dilate(gt_u8, kernel8) > 0
    ero = cv2.erode(gt_u8, kernel8) > 0
    totals['gt'] += int(gt.sum())
    totals['dil'] += int(dil.sum())
    totals['ero'] += int(ero.sum())
    totals['inter_dil'] += int((dil & gt).sum())
    totals['inter_ero'] += int((ero & gt).sum())

gt_area = totals['gt']
iou_dil = totals['inter_dil'] / (totals['dil'] + gt_area - totals['inter_dil'])
iou_ero = totals['inter_ero'] / (totals['ero'] + gt_area - totals['inter_ero'])
print(f"GT foreground pixels: {gt_area:,}  (mean per 1,983 images: {gt_area/1983:.1f} px)")
print(f"1-px dilation  area {totals['dil']:>9,}  IoU {iou_dil:.4f}")
print(f"1-px erosion   area {totals['ero']:>9,}  IoU {iou_ero:.4f}")
