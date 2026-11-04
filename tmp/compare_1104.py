from pathlib import Path
import cv2, numpy as np
new = Path(r"tmp/1104_test_maps/test_maps")
old = Path(r"Ours/test_maps")
same = diff = 0
tot_new = tot_old = 0
diffpix = 0
for name in sorted(p.name for p in new.glob("*.png"))[:20]:
    a = cv2.imread(str(new / name), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(old / name), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        continue
    tot_new += int((a > 127).sum())
    tot_old += int((b > 127).sum())
    diffpix += int((a != b).sum())
    if np.array_equal(a, b):
        same += 1
    else:
        diff += 1
print(f"first-20 identical: {same}, different: {diff}")
print(f"first-20 foreground px: new {tot_new}, old(test_maps) {tot_old}, differing px {diffpix}")
