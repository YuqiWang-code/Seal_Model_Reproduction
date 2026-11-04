"""Plot Seal dataset statistics: target number per image and target size.

Input: JSON produced by tools/compute_seal_dataset_stats.py (run on the server).
Outputs (Ours/dataset_stats/):
    seal_dataset_statistics.png            two panels (a)(b), for the paper
    seal_target_number_distribution.png    panel (a) alone
    seal_target_size_distribution.png      panel (b) alone

Both panels show the percentage on the left axis and the cumulative
percentage on the right axis (Fig. 4(b) style). White background, single-colour
bars with black edges, cumulative line + markers on a twin axis, left/bottom
spines only, no grid. All text uses Microsoft YaHei.

Binning
-------
* target number per image: exact 1..10, then 11-15 / 16-20 / >20 (95 % of the
  images have at most 10 targets; the tail reaches 47).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
EXACT = list(range(1, 11))
NUMBER_TAIL = [("11-15", 11, 15), ("16-20", 16, 20), (">20", 21, 10 ** 6)]
SIZE_EDGES = list(range(0, 60, 5))
BAR_COLOR = "#8CBF3F"
CURVE_COLOR = "#C0392B"


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
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # TrueType, accepted by most journals (not Type 3)
        "ps.fonttype": 42,
    })


def couple_axes(ax2, unit):
    ax2.set_ylim(0, 105)
    ax2.set_ylabel(f"Cumulative percent of {unit} (%)", color=CURVE_COLOR)
    ax2.tick_params(axis="y", colors=CURVE_COLOR, direction="in")
    ax2.spines["right"].set_color(CURVE_COLOR)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.yaxis.set_major_locator(MultipleLocator(20))


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(False)


def add_legend(ax, loc="upper right", anchor=None):
    handles = [Patch(facecolor=BAR_COLOR, edgecolor="black", linewidth=0.8, label="Percentage"),
               Line2D([], [], color=CURVE_COLOR, marker="o", markersize=3.5, linewidth=1.2,
                      label="Cumulative")]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=True, fontsize=8.5,
              borderpad=0.5, handlelength=1.6)


def panel_number(ax, stats):
    per_image = {int(k): v for k, v in stats["per_image"].items()}
    labels = [str(k) for k in EXACT]
    counts = [per_image.get(k, 0) for k in EXACT]
    for name, lo, hi in NUMBER_TAIL:
        labels.append(name)
        counts.append(sum(v for k, v in per_image.items() if lo <= k <= hi))
    total = sum(counts)
    pct = [c / total * 100 for c in counts]
    cum, running = [], 0.0
    for p in pct:
        running += p
        cum.append(running)

    ax2 = ax.twinx()
    x = list(range(len(counts)))
    ax.bar(x, pct, width=0.72, color=BAR_COLOR, edgecolor="black", linewidth=0.8, zorder=2)
    ax2.plot(x, cum, color=CURVE_COLOR, marker="o", markersize=3.6, linewidth=1.3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick, label in zip(ax.get_xticklabels(), labels):
        if "-" in label or label.startswith(">"):
            tick.set_fontsize(8.5)
            tick.set_rotation(32)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
    ax.set_xlim(-0.7, len(counts) - 0.3)
    ax.set_ylim(0, max(pct) * 1.18)
    ax.set_xlabel("Target number per image")
    ax.set_ylabel("Percentage of images (%)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    tidy(ax)
    couple_axes(ax2, "images")
    add_legend(ax, loc="center right", anchor=(1.0, 0.45))


def panel_size(ax, stats):
    area = {int(k): v for k, v in stats["area_hist"].items()}
    counts = [sum(v for k, v in area.items() if lo <= k < lo + 5) for lo in SIZE_EDGES]
    overflow = sum(v for k, v in area.items() if k >= SIZE_EDGES[-1] + 5)
    total = sum(counts) + overflow
    pct = [c / total * 100 for c in counts] + [overflow / total * 100]
    cum, running = [], 0.0
    for p in pct:
        running += p
        cum.append(running)

    centers = [lo + 2.5 for lo in SIZE_EDGES]
    x_tail = SIZE_EDGES[-1] + 9.5
    ax2 = ax.twinx()
    ax.bar(centers, pct[:-1], width=4.3, color=BAR_COLOR, edgecolor="black", linewidth=0.7, zorder=2)
    ax.bar([x_tail], [pct[-1]], width=4.3, color=BAR_COLOR, edgecolor="black", linewidth=0.7, zorder=2)
    ax2.plot(centers + [x_tail], cum, color=CURVE_COLOR, marker="o", markersize=3.4,
             linewidth=1.3, zorder=4)

    ticks = list(range(0, SIZE_EDGES[-1] + 6, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.annotate(">60", xy=(x_tail, 0), xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=9.5, annotation_clip=False)
    ax.set_xlim(0, x_tail + 4.5)
    ax.set_ylim(0, max(pct) * 1.18)
    ax.set_xlabel("Target size (pixels)")
    ax.set_ylabel("Percentage of targets (%)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    tidy(ax)
    couple_axes(ax2, "targets")
    add_legend(ax, loc="center right", anchor=(1.0, 0.62))


def label_panels(axes):
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.annotate(tag, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -46),
                    textcoords="offset points", ha="center", va="top", fontsize=11)


def render(stats, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1))
    panel_number(axes[0], stats)
    panel_size(axes[1], stats)
    label_panels(axes)
    fig.subplots_adjust(wspace=0.45, bottom=0.22)
    out = outdir / "seal_dataset_statistics.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")

    for maker, name in ((panel_number, "seal_target_number_distribution.png"),
                        (panel_size, "seal_target_size_distribution.png")):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        maker(ax, stats)
        fig.subplots_adjust(bottom=0.16)
        out = outdir / name
        fig.savefig(out)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"wrote {out} (+ .pdf)")


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
    render(stats, args.outdir)
    print("images", stats["images"], "targets", stats["targets"],
          "empty", stats["empty_images"], "area", stats["area_summary"])


if __name__ == "__main__":
    main()
