from pathlib import Path
import cv2
pred = Path(r"tmp/1104_test_maps/test_maps")
gt = Path(r"Ours/Seal_test_GT")
pn = {p.name for p in pred.glob("*.png")}
gn = {p.name for p in gt.glob("*.png")}
print("pred", len(pn), "gt", len(gn), "missing", len(gn - pn), "extra", len(pn - gn))
sizes = {}
bad = []
for name in sorted(pn)[:40]:
    p = cv2.imread(str(pred / name), cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(str(gt / name), cv2.IMREAD_GRAYSCALE)
    sizes[p.shape] = sizes.get(p.shape, 0) + 1
    if p.shape != g.shape:
        bad.append((name, p.shape, g.shape))
print("first-40 prediction shapes:", sizes)
print("shape mismatches in first 40:", bad[:5])
vals = set()
for name in sorted(pn)[:20]:
    p = cv2.imread(str(pred / name), cv2.IMREAD_GRAYSCALE)
    vals.update([int(p.min()), int(p.max())])
print("first-20 value range:", min(vals), max(vals))
