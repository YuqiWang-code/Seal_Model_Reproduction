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

import figure_style

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


def round_polyline(xs, ys, radius, samples=14):
    """Fillet every interior corner with a quadratic Bezier (rounded elbows).

    ``radius`` is in axis units (both axes are 0-1, so one number works).  A corner
    whose adjacent segments are shorter than the radius automatically gets a smaller
    fillet (the tangent offsets are clamped to half a segment), so dense measured
    points stay intact and only the long straight runs get visibly softened.
    """
    pts = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    if radius <= 0 or pts.shape[0] < 3:
        return pts[:, 0], pts[:, 1]
    out = [pts[0]]
    for i in range(1, pts.shape[0] - 1):
        a, v, b = pts[i - 1], pts[i], pts[i + 1]
        va, vb = a - v, b - v
        la, lb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if la < 1e-9 or lb < 1e-9:
            out.append(v)
            continue
        ta, tb = min(radius / la, 0.5), min(radius / lb, 0.5)
        p1, p2 = v + va * ta, v + vb * tb
        out.append(p1)
        for t in np.linspace(0.0, 1.0, samples + 2)[1:-1]:
            out.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * v + t ** 2 * p2)
        out.append(p2)
    out.append(pts[-1])
    arr = np.asarray(out)
    return arr[:, 0], arr[:, 1]


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
    ap.add_argument('--show-ap', dest='show_ap', action='store_true',
                    help='append "[AP=0.xxx]" to each legend entry')
    ap.add_argument('--no-show-ap', dest='show_ap', action='store_false')
    ap.set_defaults(show_ap=False)
    ap.add_argument('--font-scale', type=float, default=1.0,
                    help='multiply every font size by this')
    ap.add_argument('--corner-radius', type=float, default=0.018,
                    help='fillet radius in axis units for the elbows (0 = sharp corners)')
    ap.add_argument('--close', dest='close', action='store_true',
                    help='draw the dashed closing segment to (1, 0)')
    ap.add_argument('--no-close', dest='close', action='store_false')
    ap.set_defaults(close=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding='utf-8'))
    items = man['methods'] if isinstance(man, dict) else man
    os.makedirs(args.outdir, exist_ok=True)
    cmap = color_map([m['label'] for m in items], args.highlight)

    fs = args.font_scale
    figure_style.apply(fs)
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
        # 整条线一条实线画完（闭合段也实线），不画点；把每个拐角做圆倒角
        xs, ys = round_polyline(xs, ys, args.corner_radius)
        ax.plot(xs, ys, color=cmap[m['label']], solid_capstyle='round',
                solid_joinstyle='round',
                lw=2.1 if main_ else 1.15, zorder=6 if main_ else 3,
                label=f"{m['label']}  [AP={score:.3f}]" if args.show_ap else m['label'])

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    figure_style.axis_labels(ax, 'Recall', 'Precision', fs)
    ax.set_xticks(np.linspace(0, 1, 6))
    figure_style.finish(ax, fs, y_ticks=np.linspace(0, 1, 6))
    if args.title:
        ax.set_title(args.title, fontsize=9.5 * fs)
    ax.legend(fontsize=figure_style.LEGEND_PT * fs, ncol=args.legend_ncol, loc=args.legend_loc,
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
