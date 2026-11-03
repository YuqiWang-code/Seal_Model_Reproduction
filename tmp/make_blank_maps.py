"""Create an all-zero prediction folder matching the Seal test GT names.

Used to measure the trivial (predict-everything-background) baseline of the
four original metrics, so the source-table values can be read against it.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

gt_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

count = 0
for gt_path in sorted(gt_dir.glob('*.png')):
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise SystemExit(f'cannot read {gt_path}')
    out = out_dir / gt_path.name
    if not cv2.imwrite(str(out), np.zeros(gt.shape, dtype=np.uint8)):
        raise SystemExit(f'cannot write {out}')
    count += 1
print(f'wrote {count} blank predictions to {out_dir}')
