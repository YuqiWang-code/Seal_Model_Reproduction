#!/usr/bin/env python3
"""Textbook-style pixel-level Precision-Recall figure (sklearn convention).

Differences from tools/plot_roc_pr.py's P-R panel, all of them display choices that
follow the standard ``sklearn.metrics.precision_recall_curve`` convention:

* every 8-bit threshold is drawn, **including** ``t = 0`` ("predict everything"),
  so the polyline ends at ``(recall = 1, precision = foreground rate)`` -- exactly
  where sklearn puts its last point. No extrapolation is needed;
* the (0, 1) sentinel is optional (``--sentinel/--no-sentinel``);
* points are drawn as small markers every ``--markevery`` vertices, like the
  textbook figures this replaces;
* the AP is printed in the legend.

Usage:
    python tools/plot_pr_textbook.py --manifest tmp/roc/manifest_v6.json \
        --outdir tmp/roc/pr_textbook --stem pr_2 --no-sentinel
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


def pr_full(rec):
    """Precision/recall at every 8-bit threshold, sklearn order (recall ascending).

    ``t = 0`` is 'predict everything': recall 1, precision = n_pos / (n_pos + n_neg).
    Keeping it is what makes the curve end in the bottom-right corner with a real
    point instead of an extrapolated segment.
    """
    pos = np.asarray(rec['pos_hist'], dtype=np.float64)
    neg = np.asarray(rec['neg_hist'], dtype=np.float64)
    n_pos, n_neg = pos.sum(), neg.sum()
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    rec_ = tp / max(n_pos, 1.0)
    prec_ = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    order = np.argsort(rec_, kind='stable')       # t = 255 ... 0
    r, p = rec_[order], prec_[order]
    # collapse duplicate recall values (8-bit ties): keep the largest precision
    keep = np.ones(r.size, dtype=bool)
    for k in range(r.size - 2, -1, -1):
        if r[k] == r[k + 1]:
            keep[k] = False
    return r[keep], p[keep]


def ap_sklearn(rec, prec):
    """AP = sum (R_n - R_{n-1}) * P_n, sklearn's step interpolation."""
    return float(np.sum(np.diff(rec) * prec[1:]))


def schematic(rec, prec, until='end', frac=0.8):
    """Two-segment schematic of a PR curve: flat at the best precision, then a plunge.

    The reference textbook figures stay at precision ~1 until recall saturates and
    only then drop; 8-bit Seal maps never reach precision 1, so every method gets
    its OWN best precision (the highest measured point) as the plateau height.
    ``until='end'``  : the plateau runs to the last measured recall (the method's
                       reachable recall) and then drops straight to (1, 0);
    ``until='frac'`` : the plateau runs to the first recall where the measured
                       precision falls below ``frac`` x best, then drops to (1, 0).
    """
    r_max = float(np.max(rec))
    p_best = float(np.max(prec))
    if until == 'end':
        r_knee = r_max
    else:
        hit = np.flatnonzero(prec < frac * p_best)
        r_knee = float(rec[hit[0]]) if hit.size else r_max
    return np.array([0.0, r_knee, 1.0]), np.array([p_best, p_best, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--stem', default='pr_2')
    ap.add_argument('--title', default='')
    ap.add_argument('--fig-width', type=float, default=6.0)
    ap.add_argument('--fig-height', type=float, default=5.0)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--markevery', type=int, default=8,
                    help='draw a marker every N vertices (0 = no markers)')
    ap.add_argument('--sentinel', dest='sentinel', action='store_true',
                    help='prepend the classic (0, 1) point')
    ap.add_argument('--no-sentinel', dest='sentinel', action='store_false')
    ap.set_defaults(sentinel=False)
    ap.add_argument('--legend-loc', default='lower left')
    ap.add_argument('--legend-ncol', type=int, default=1)
    ap.add_argument('--schematic', choices=('none', 'end', 'frac'), default='none',
                    help='draw a two-segment schematic (flat best-precision plateau, then '
                         'a plunge to (1,0)) instead of the measured polyline')
    ap.add_argument('--plateau-frac', type=float, default=0.8,
                    help='with --schematic frac: plateau ends where precision < frac x best')
    ap.add_argument('--highlight', default='Ours')
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding='utf-8'))
    methods = man['methods'] if isinstance(man, dict) else man
    os.makedirs(args.outdir, exist_ok=True)
    cmap = color_map([m['label'] for m in methods], args.highlight)

    plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                         'xtick.direction': 'out', 'ytick.direction': 'out'})
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    stats = []
    for m in methods:
        rec = json.load(open(m['hist'], encoding='utf-8'))
        r, p = pr_full(rec)
        score = ap_sklearn(np.concatenate([[0.0], r]), np.concatenate([[1.0], p]))
        stats.append((m['label'], score))
        main_ = m['label'] == args.highlight
        if args.schematic != 'none':
            r, p = schematic(r, p, args.schematic, args.plateau_frac)
            ax.plot(r, p, color=cmap[m['label']],
                    lw=2.1 if main_ else 1.15, ls='-' if main_ else (0, (5, 1.6)),
                    zorder=6 if main_ else 3,
                    label=f"{m['label']}  [AP={score:.3f}]")
            continue
        if args.sentinel:
            r = np.concatenate([[0.0], r])
            p = np.concatenate([[1.0], p])
        ax.plot(r, p, color=cmap[m['label']],
                lw=2.1 if main_ else 1.15,
                marker='o' if args.markevery else None,
                ms=(3.0 if main_ else 2.2) if args.markevery else 0,
                markevery=args.markevery,
                mfc='white' if args.markevery else None,
                mew=0.8 if args.markevery else 0,
                zorder=6 if main_ else 3,
                label=f"{m['label']}  [AP={score:.3f}]")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, which='major', ls='-', lw=0.45, color='0.86', zorder=0)
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
    for label, score in stats:
        print(f'{label:12s} AP={score:.4f}')
    print(f'wrote {png} / {pdf}')


if __name__ == '__main__':
    main()
