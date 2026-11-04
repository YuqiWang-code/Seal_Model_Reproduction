"""Plot Seal dataset statistics (target number per image, target size).

Input: JSON produced by tmp/seal_dataset_stats.py (run on the server).
Output: two PNGs in Ours/dataset_stats/.

Style follows the IRSTD dataset-statistics figure used in papers: white
background, single-colour bars with black edges, left/bottom spines only, no
grid. All text uses Microsoft YaHei as requested.

Binning
-------
* target number per image: exact 1..10, then 11-15 / 16-20 / >20, because
  95 % of the images have at most 10 targets and the tail reaches 47.
* target size: 5-pixel bins from 0 to 60, plus one ">60" bar; only 30 of the
  12,554 targets (0.24 %) are larger than 60 px (max 114).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]

EXACT = list(range(1, 11))
NUMBER_TAIL = [("11-15", 11, 15), ("16-20", 16, 20), (">20", 21, 10 ** 6)]
SIZE_EDGES = list(range(0, 60, 5))
BAR_COLOR = "#8CBF3F"


def pick_font():
    want = ["Microsoft YaHei", "Microsoft YaHei UI", "msyh", "SimHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in want:
        if name in have:
            return name
    return "DejaVu Sans"


def setup(font):
    plt.rcParams.update({
        "font.family": font,
        "font.size": 10,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def finish(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(False)


def bar_target_number(stats, outdir: Path):
    per_image = {int(k): v for k, v in stats["per_image"].items()}
    labels, counts = [str(k) for k in EXACT], [per_image.get(k, 0) for k in EXACT]
    for name, lo, hi in NUMBER_TAIL:
        labels.append(name)
        counts.append(sum(v for k, v in per_image.items() if lo <= k <= hi))

    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    x = list(range(len(counts)))
    ax.bar(x, counts, width=0.72, color=BAR_COLOR, edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick, label in zip(ax.get_xticklabels(), labels):
        if "-" in label or label.startswith(">"):
            tick.set_fontsize(9)
    ax.set_xlabel("Target number per image")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.14)
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.set_xlim(-0.7, len(counts) - 0.3)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.012, f"{c}", ha="center", va="bottom", fontsize=8.5)
    finish(ax)
    out = outdir / "seal_target_number_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  images {sum(counts)}")


def bar_target_size(stats, outdir: Path):
    area = {int(k): v for k, v in stats["area_hist"].items()}
    counts = [sum(v for k, v in area.items() if lo <= k < lo + 5) for lo in SIZE_EDGES]
    overflow = sum(v for k, v in area.items() if k >= SIZE_EDGES[-1] + 5)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    centers = [lo + 2.5 for lo in SIZE_EDGES]
    ax.bar(centers, counts, width=4.3, color=BAR_COLOR, edgecolor="black", linewidth=0.7, zorder=3)
    x_tail = SIZE_EDGES[-1] + 9.5
    ax.bar([x_tail], [overflow], width=4.3, color=BAR_COLOR, edgecolor="black", linewidth=0.7, zorder=3)
    ticks = list(range(0, SIZE_EDGES[-1] + 6, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.annotate(">60", xy=(x_tail, 0), xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=10, annotation_clip=False)
    ax.set_xlim(0, x_tail + 4)
    ax.set_xlabel("Target size (pixels)")
    ax.set_ylabel("Number of targets")
    ax.set_ylim(0, max(counts) * 1.14)
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    for center, count in list(zip(centers, counts)) + [(x_tail, overflow)]:
        if count:
            ax.text(center, count + max(counts) * 0.012, f"{count}", ha="center", va="bottom", fontsize=8)
    finish(ax)
    out = outdir / "seal_target_size_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  targets {sum(counts) + overflow} (>{SIZE_EDGES[-1]}: {overflow})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path,
                    default=ROOT / "Ours" / "dataset_stats" / "seal_dataset_stats.json")
    ap.add_argument("--outdir", type=Path, default=ROOT / "Ours" / "dataset_stats")
    ap.add_argument("--split", default="all", help="all / train / val / test")
    args = ap.parse_args()

    data = json.loads(args.stats.read_text(encoding="utf-8"))
    stats = data[args.split]
    args.outdir.mkdir(parents=True, exist_ok=True)
    font = pick_font()
    print("font:", font)
    setup(font)
    bar_target_number(stats, args.outdir)
    bar_target_size(stats, args.outdir)
    print("images", stats["images"], "targets", stats["targets"],
          "empty", stats["empty_images"], "area", stats["area_summary"])


if __name__ == "__main__":
    main()
