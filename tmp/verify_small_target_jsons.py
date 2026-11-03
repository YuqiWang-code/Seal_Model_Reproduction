"""Recount IoU/F1/Pd/Fa from the stored counts and compare with stored metrics."""
import glob
import json
import os

ROOT = r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction"

targets = sorted(glob.glob(os.path.join(ROOT, "docs", "小目标指标评估_20261103", "*.json")))
targets += [os.path.join(ROOT, "docs", "NESS-Net_Seal评估产物", "refinement_namlab_small_target_metrics.json"),
            os.path.join(ROOT, "docs", "NESS-Net_Seal评估产物", "small_target_metrics.json")]

records = []
for path in targets:
    data = json.load(open(path, encoding="utf-8"))
    if "counts" in data:
        records.append((os.path.basename(path), data))
    else:  # summary.json holds per-method entries
        for name, entry in data.items():
            records.append((f"summary.json:{name}", entry))

print(f"{'source':40s} {'IoU':>10s} {'IoU*':>10s} {'F1':>10s} {'F1*':>10s} {'Pd':>10s} {'Pd*':>10s} {'Fa':>12s} {'Fa*':>12s}")
for name, e in records:
    c, m = e["counts"], e["metrics"]
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    union = tp + fp + fn
    f1d = 2 * tp + fp + fn
    iou = tp / union if union else 0.0
    f1 = 2 * tp / f1d if f1d else 0.0
    pd = c["n_detected"] / c["n_gt"] if c["n_gt"] else 0.0
    fa = c["false_alarm_pixels"] / c["total_pixels"] if c["total_pixels"] else 0.0
    flags = []
    for key, mine in (("IoU", iou), ("F1", f1), ("Pd", pd), ("Fa", fa)):
        if abs(m[key] - mine) > 1e-12:
            flags.append(f"{key} differs by {m[key]-mine:+.3e}")
    # extra checks
    if c["n_pred"] < c["n_detected"]:
        flags.append("n_detected > n_pred")
    if c["false_alarm_pixels"] > tp + fp:
        flags.append("false_alarm_pixels > pred area")
    print(f"{name:40s} {m['IoU']:10.8f} {iou:10.8f} {m['F1']:10.8f} {f1:10.8f} {m['Pd']:10.8f} {pd:10.8f} {m['Fa']:12.10f} {fa:12.10f}  {'; '.join(flags)}")
