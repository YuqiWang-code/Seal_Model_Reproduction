"""Plot Seal dataset statistics: target number per image and target size.

Input: JSON produced by tools/compute_seal_dataset_stats.py (run on the server).
Outputs (Ours/dataset_stats/):
    seal_dataset_statistics.png / .pdf     two panels (a)(b), paper-ready
    seal_target_number_distribution.png/.pdf   panel (a) alone
    seal_target_size_distribution.png/.pdf     panel (b) alone

Layout is designed at print size (7.2 in wide, i.e. one full text width of a
two-column paper), so the PDF points are the final point sizes: bars show the
percentage on the left axis, a light cumulative curve sits on the right axis
(Fig. 4(b) style). White background, muted palette, hairline axes, faint
horizontal grid, no chart junk. All text uses Microsoft YaHei.

Binning
-------
* target number per image: exact 1..10, then one ">=11" bar separated by a gap
  (95 % of the images have at most 10 targets; the tail reaches 47).
* target size: 5-pixel bins from 0 to 60. Only 30 of the 12,554 targets
  (0.24 %, max 114 px) are larger than 60 px; they are not drawn as a bar but
  reported in a corner note, so the axis stays honest and uncluttered.
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
SIZE_EDGES = list(range(0, 60, 5))

BAR_FILL = "#A7CB55"
BAR_EDGE = "#5B6B3A"
CURVE_COLOR = "#C0492F"
GRID_COLOR = "#E8E8E8"
TEXT_DARK = "#333333"

FIG_W = 7.8          # inches; a two-column paper prints it at ~7 in
PANEL_H = 2.75
FONT = None


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
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": "#4A4A4A",
        "text.color": TEXT_DARK,
        "axes.labelcolor": TEXT_DARK,
        "xtick.color": "#4A4A4A",
        "ytick.color": "#4A4A4A",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def couple(ax2):
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative percentage (%)", color=CURVE_COLOR, fontsize=8)
    ax2.tick_params(axis="y", colors=CURVE_COLOR, labelsize=7.5, length=2.5, width=0.6)
    ax2.spines["right"].set_color(CURVE_COLOR)
    ax2.spines["right"].set_linewidth(0.7)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.grid(False)


def cumulative(values):
    out, run = [], 0.0
    for v in values:
        run += v
        out.append(run)
    return out


def style_bars(ax, xs, values, width):
    return ax.bar(xs, values, width=width, color=BAR_FILL, edgecolor=BAR_EDGE,
                  linewidth=0.6, zorder=2)


def panel_number(ax, stats, width_scale=1.0):
    per_image = {int(k): v for k, v in stats["per_image"].items()}
    total = sum(per_image.values())
    labels = [str(k) for k in EXACT]
    counts = [per_image.get(k, 0) for k in EXACT]
    tail = sum(v for k, v in per_image.items() if k >= 11)
    labels.append("≥11")
    counts.append(tail)

    xs = list(range(len(EXACT))) + [len(EXACT) + 0.6]
    pct = [c / total * 100 for c in counts]

    ax2 = ax.twinx()
    style_bars(ax, xs, pct, 0.7 * width_scale)
    ax2.plot(xs, cumulative(pct), color=CURVE_COLOR, marker="o", markersize=2.6,
             markerfacecolor=CURVE_COLOR, markeredgecolor="white", markeredgewidth=0.4,
             linewidth=1.0, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Target number per image")
    ax.set_ylabel("Percentage of images (%)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    tidy(ax)
    couple(ax2)


def panel_size(ax, stats, width_scale=1.0):
    area = {int(k): v for k, v in stats["area_hist"].items()}
    counts = [sum(v for k, v in area.items() if lo <= k < lo + 5) for lo in SIZE_EDGES]
    total = sum(area.values())
    overflow = sum(v for k, v in area.items() if k >= SIZE_EDGES[-1] + 5)
    pct = [c / total * 100 for c in counts]

    centers = [lo + 2.5 for lo in SIZE_EDGES]
    ax2 = ax.twinx()
    style_bars(ax, centers, pct, 4.0 * width_scale)
    ax2.plot(centers, cumulative(pct), color=CURVE_COLOR, marker="o", markersize=2.6,
             markerfacecolor=CURVE_COLOR, markeredgecolor="white", markeredgewidth=0.4,
             linewidth=1.0, zorder=4)

    ax.set_xticks(list(range(0, 61, 10)))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Target size (pixels)")
    ax.set_ylabel("Percentage of targets (%)")
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    tidy(ax)
    couple(ax2)
    ax.annotate(f">{SIZE_EDGES[-1] + 5} px: {overflow} targets ({overflow / total * 100:.2f} %)",
                xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.6, color="#6B6B6B")


def shared_legend(fig):
    handles = [Patch(facecolor=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.6, label="Percentage"),
               Line2D([], [], color=CURVE_COLOR, marker="o", markersize=2.6,
                      markerfacecolor=CURVE_COLOR, markeredgecolor="white",
                      markeredgewidth=0.4, linewidth=1.0, label="Cumulative")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02),
               ncol=2, frameon=False, fontsize=7.8, handlelength=1.4,
               columnspacing=1.6, handletextpad=0.5)


def render_combined(stats, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H))
    panel_number(axes[0], stats)
    panel_size(axes[1], stats)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.annotate(tag, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.235, top=0.86, wspace=0.42)
    shared_legend(fig)
    out = outdir / "seal_dataset_statistics.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def render_single(stats, outdir: Path, maker, name):
    fig, ax = plt.subplots(figsize=(3.8, 2.75))
    maker(ax, stats)
    ax.annotate("", xy=(0, 0))
    fig.subplots_adjust(left=0.155, right=0.855, bottom=0.24, top=0.94)
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
    render_combined(stats, args.outdir)
    render_single(stats, args.outdir, panel_number, "seal_target_number_distribution.png")
    render_single(stats, args.outdir, panel_size, "seal_target_size_distribution.png")
    print("images", stats["images"], "targets", stats["targets"],
          "empty", stats["empty_images"], "area", stats["area_summary"])


if __name__ == "__main__":
    main()
