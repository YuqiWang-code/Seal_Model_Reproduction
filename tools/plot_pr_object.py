#!/usr/bin/env python3
"""Object-level Precision-Recall figure (the textbook / YOLO-style P-R curve).

Reads the ``tools/target_pr.py`` JSONs (one object per GT target / predicted
component, 8-connected + centroid <3 px one-to-one matching) and draws

    x = recall   = matched GT targets / all GT targets        (= Pd in the tables)
    y = precision = matched prediction components / all reported components

which is the object-level analogue of the confidence-threshold P-R curve in the
reference figures: precision starts high (most reported blobs are real targets) and
only the low thresholds add noise, so the curve is a plateau with a drop at the right
instead of the pixel-level "declining chord" that 8-bit maps produce.

The curve is drawn like the textbook ones: the (0, 1) convention point, one polyline
through the measured thresholds (ordered by recall, cut where merging components make
recall fall again), then a dashed closing segment to (1, 0) -- labeled as a convention
because on Seal a few percent of the targets are never detected at any threshold.

Usage:
    python tools/plot_pr_object.py --manifest tmp/roc/manifest_target_pr.json \
        --outdir Ours/roc_pr/v13_20261105 --stem pr_obj
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MAIN_COLOR = '#d62728'
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
          '#17becf', '#bcbd22', '#7f7f7f', '#393b79', '#637939', '#8c6d31',
          '#4e79a7', '#a55194', '#76b7b2', '#59a14f']


def color_map(labels, highlight):
    out, j = {}, 0
    for label in labels:
        if label == highlight:
            out[label] = MAIN_COLOR
        else:
            out[label] = COLORS[j % len(COLORS)]
            j += 1
    return out


def object_curve(rec, envelope=True):
    """Recall-ordered points, cut at the recall maximum (merging blobs beyond it).

    Thresholds that predict **no component at all** are dropped: that operating point
    is "report nothing" -- precision is undefined and the (0, 1) convention point at
    the start of the curve already stands for it.  Leaving them in put a spurious
    (0, 0) vertex at the left edge for the three methods whose maps are empty at
    ``t = 255`` (UIU-Net / SCTransNet / ILNet).
    """
    r = np.asarray(rec['recall'], dtype=np.float64)
    p = np.asarray(rec['precision'], dtype=np.float64)
    t = np.asarray(rec['thresholds'], dtype=np.float64)
    n = np.asarray(rec['n_pred_components'], dtype=np.float64)
    keep = n > 0
    r, p, t = r[keep], p[keep], t[keep]
    order = np.argsort(-t)                      # high threshold first = low recall
    r, p = r[order], p[order]
    sel = [0]
    for k in range(1, r.size):
        if r[k] > r[sel[-1]]:
            sel.append(k)
    r, p = r[sel], p[sel]
    if envelope:
        # 上包络（sklearn 的 AP 口径）：precision(r) = 所有 recall >= r 的点里的最大精度。
        # 这样曲线单调不升、不会因个别阈值的抖动回勾，形状与论文里的 P-R 一致。
        p = np.maximum.accumulate(p[::-1])[::-1]
    return r, p


def ap_step(rec, prec):
    """AP with the (0,1) sentinel and the (1,0) closing point (sklearn-style steps)."""
    r = np.concatenate([[0.0], rec, [1.0]])
    p = np.concatenate([[1.0], prec, [0.0]])
    return float(np.sum(np.diff(r) * p[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True,
                    help='JSON list of {"label": ..., "pr": path} entries')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--stem', default='pr_obj')
    ap.add_argument('--title', default='')
    ap.add_argument('--fig-width', type=float, default=6.2)
    ap.add_argument('--fig-height', type=float, default=4.9)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--highlight', default='Ours')
    ap.add_argument('--legend-loc', default='lower left')
    ap.add_argument('--legend-ncol', type=int, default=2)
    ap.add_argument('--close', dest='close', action='store_true',
                    help='draw the dashed closing segment to (1, 0)')
    ap.add_argument('--no-close', dest='close', action='store_false')
    ap.set_defaults(close=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding='utf-8'))
    items = man['methods'] if isinstance(man, dict) else man
    os.makedirs(args.outdir, exist_ok=True)
    cmap = color_map([m['label'] for m in items], args.highlight)

    plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    rows = []
    for m in items:
        rec = json.load(open(m['pr'], encoding='utf-8'))
        r, p = object_curve(rec)
        score = ap_step(r, p)
        rows.append((m['label'], float(r[-1]), float(p[-1]), score))
        main_ = m['label'] == args.highlight
        xs = np.concatenate([[0.0], r])
        ys = np.concatenate([[1.0], p])
        if args.close:
            xs = np.concatenate([xs, [1.0]])
            ys = np.concatenate([ys, [0.0]])
            ax.plot(xs[:-1], ys[:-1], color=cmap[m['label']],
                    lw=2.1 if main_ else 1.15, marker='o',
                    ms=3.0 if main_ else 2.3, mfc='white', mew=0.85,
                    zorder=6 if main_ else 3,
                    label=f"{m['label']}  [AP={score:.3f}]")
            ax.plot([xs[-2], 1.0], [ys[-2], 0.0], color=cmap[m['label']],
                    lw=1.5 if main_ else 0.9, ls=(0, (3.2, 2.0)), alpha=0.9,
                    zorder=5 if main_ else 2)
        else:
            ax.plot(xs, ys, color=cmap[m['label']], lw=2.1 if main_ else 1.15,
                    marker='o', ms=3.0 if main_ else 2.3, mfc='white', mew=0.85,
                    zorder=6 if main_ else 3,
                    label=f"{m['label']}  [AP={score:.3f}]")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, ls='-', lw=0.45, color='0.86', zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if args.title:
        ax.set_title(args.title, fontsize=9.5)
    ax.legend(fontsize=6.3, ncol=args.legend_ncol, loc=args.legend_loc,
              frameon=True, edgecolor='0.85', framealpha=0.93,
              labelspacing=0.26, handlelength=1.6, borderpad=0.4)

    png = os.path.join(args.outdir, f'{args.stem}.png')
    pdf = os.path.join(args.outdir, f'{args.stem}.pdf')
    fig.savefig(png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'{"method":12s} {"recall_end":>10s} {"prec_end":>9s} {"AP":>7s}')
    for label, r_end, p_end, score in sorted(rows, key=lambda t: -t[1]):
        print(f'{label:12s} {r_end:10.4f} {p_end:9.4f} {score:7.4f}')
    print(f'wrote {png} / {pdf}')


if __name__ == '__main__':
    main()
