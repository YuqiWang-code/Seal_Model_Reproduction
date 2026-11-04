"""Plot Seal dataset statistics: target number per image and target size.

Input: JSON produced by tools/compute_seal_dataset_stats.py (run on the server).
Outputs (Ours/dataset_stats/):
    seal_dataset_statistics.png / .pdf         two panels (a)(b), paper-ready
    seal_target_number_distribution.png / .pdf panel (a) alone
    seal_target_size_distribution.png / .pdf   panel (b) alone

Layout is designed at print size (7.8 in wide, i.e. one text width of a
two-column paper) so the PDF points are the final point sizes. Panel (b) is
1.15x wider than (a) because its long tail needs the room. Bars are deep blue,
the cumulative curve (right axis) light blue; white background, hairline axes,
faint horizontal grid, no chart junk. All text uses Microsoft YaHei.

Binning
-------
* target number per image: only 1..10 are drawn (95.2 % of the images); the
  remaining 190 images (4.8 %) have 11-47 targets and are reported in a corner
  note instead of one squeezed ">=11" bar.
* target size: 5-pixel bins from 0 to 60. Only 30 of the 12,554 targets
  (0.24 %, max 114 px) are larger than 60 px; they are reported in a corner
  note, so the axis stays honest and uncluttered.
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

BAR_FILL = "#2E5E8C"       # deep blue
BAR_EDGE = "#1E3F5E"
CURVE_COLOR = "#6FA8D0"    # light blue
CURVE_TEXT = "#4A7FA8"     # readable shade for the right axis text
GRID_COLOR = "#EAEAEA"
TEXT_DARK = "#333333"

FIG_W = 7.8
PANEL_H = 2.75
WIDTH_RATIOS = (1.0, 1.15)


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


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))


def couple(ax2):
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative percentage (%)", color=CURVE_TEXT, fontsize=8)
    ax2.tick_params(axis="y", colors=CURVE_TEXT, labelsize=7.5, length=2.5, width=0.6)
    ax2.spines["right"].set_color(CURVE_COLOR)
    ax2.spines["right"].set_linewidth(0.8)
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


def draw_curve(ax2, xs, pct):
    ys = cumulative(pct)
    # a slightly wider white line under the coloured one keeps the light curve
    # readable where it crosses the dark bars, without a halo around the markers
    ax2.plot(xs, ys, color="white", linewidth=2.2, solid_capstyle="round", zorder=3.5)
    ax2.plot(xs, ys, color=CURVE_COLOR, marker="o", markersize=2.6,
             markerfacecolor=CURVE_COLOR, markeredgecolor="white", markeredgewidth=0.4,
             linewidth=1.35, zorder=4)


def panel_number(ax, stats):
    per_image = {int(k): v for k, v in stats["per_image"].items()}
    total = sum(per_image.values())
    counts = [per_image.get(k, 0) for k in EXACT]
    pct = [c / total * 100 for c in counts]
    tail = total - sum(counts)
    xs = list(range(len(counts)))

    ax2 = ax.twinx()
    ax.bar(xs, pct, width=0.68, color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.6, zorder=2)
    draw_curve(ax2, xs, pct)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in EXACT])
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    ax.set_xlabel("Target number per image")
    ax.set_ylabel("Percentage of images (%)")
    tidy(ax)
    couple(ax2)
    ax.annotate(f">10 targets: {tail} images ({tail / total * 100:.1f} %), see text",
                xy=(0.97, 0.47), xycoords="axes fraction", ha="right", va="top",
                fontsize=6.6, color="#6B6B6B")


def panel_size(ax, stats):
    area = {int(k): v for k, v in stats["area_hist"].items()}
    counts = [sum(v for k, v in area.items() if lo <= k < lo + 5) for lo in SIZE_EDGES]
    total = sum(area.values())
    overflow = sum(v for k, v in area.items() if k >= SIZE_EDGES[-1] + 5)
    pct = [c / total * 100 for c in counts]

    centers = [lo + 2.5 for lo in SIZE_EDGES]
    ax2 = ax.twinx()
    ax.bar(centers, pct, width=4.0, color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.55, zorder=2)
    draw_curve(ax2, centers, pct)

    ax.set_xticks(list(range(0, 61, 10)))
    ax.set_xlim(0, 60)
    ax.set_xlabel("Target size (pixels)")
    ax.set_ylabel("Percentage of targets (%)")
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    tidy(ax)
    couple(ax2)
    ax.annotate(f">{SIZE_EDGES[-1] + 5} px: {overflow} targets ({overflow / total * 100:.2f} %)",
                xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.6, color="#6B6B6B")


def shared_legend(fig):
    handles = [Patch(facecolor=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.6, label="Percentage"),
               Line2D([], [], color=CURVE_COLOR, marker="o", markersize=2.6,
                      markerfacecolor=CURVE_COLOR, markeredgecolor="white",
                      markeredgewidth=0.4, linewidth=1.2, label="Cumulative")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02),
               ncol=2, frameon=False, fontsize=7.8, handlelength=1.4,
               columnspacing=1.6, handletextpad=0.5)


def render_combined(stats, outdir: Path):
    fig = plt.figure(figsize=(FIG_W, PANEL_H))
    gs = fig.add_gridspec(1, 2, width_ratios=WIDTH_RATIOS,
                          left=0.072, right=0.925, bottom=0.235, top=0.86, wspace=0.40)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel_number(ax_a, stats)
    panel_size(ax_b, stats)
    for ax, tag in zip((ax_a, ax_b), ("(a)", "(b)")):
        ax.annotate(tag, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9, fontweight="bold")
    shared_legend(fig)
    out = outdir / "seal_dataset_statistics.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def render_single(stats, outdir: Path, maker, name, width=3.8):
    fig, ax = plt.subplots(figsize=(width, PANEL_H))
    maker(ax, stats)
    fig.subplots_adjust(left=0.145 if width < 4 else 0.135, right=0.855, bottom=0.235, top=0.94)
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
    render_single(stats, args.outdir, panel_number, "seal_target_number_distribution.png", width=3.9)
    render_single(stats, args.outdir, panel_size, "seal_target_size_distribution.png", width=4.2)
    print("images", stats["images"], "targets", stats["targets"],
          "empty", stats["empty_images"], "area", stats["area_summary"])


if __name__ == "__main__":
    main()
