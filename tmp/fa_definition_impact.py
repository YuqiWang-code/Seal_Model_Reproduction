"""Compare the project's Fa with the pixel-FP Fa used by docs/小目标评价指标.pdf."""
import json
import glob
import os

ROOT = r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction"
files = sorted(glob.glob(os.path.join(ROOT, "docs", "小目标指标评估_20261103", "*.json")))
files.append(os.path.join(ROOT, "docs", "NESS-Net_Seal评估产物",
                          "refinement_namlab_small_target_metrics.json"))

print(f"{'source':28s} {'fp':>12s} {'unmatched':>12s} {'Fa(project)':>14s} {'Fa(pixelFP)':>14s} {'ratio':>8s}")
for path in files:
    data = json.load(open(path, encoding="utf-8"))
    entries = [("this", data)] if "counts" in data else list(data.items())
    for name, e in entries:
        c = e["counts"]
        fp, total = c["fp"], c["total_pixels"]
        fa_project = c["false_alarm_pixels"] / total
        fa_pixel = fp / total
        print(f"{os.path.basename(path)[:20] + ':' + name[:7]:28s} {fp:12d} {c['false_alarm_pixels']:12d} "
              f"{fa_project:14.10f} {fa_pixel:14.10f} {fa_pixel / fa_project if fa_project else 0:7.1f}x")
