#!/usr/bin/env python3
"""Draw pixel-level ROC / PR curves from the histograms written by tools/roc_hist.py.

One method = one ``<name>.json`` with the 256-bin foreground/background prediction
histograms. Every figure is saved twice (600 dpi PNG + vector PDF) together with a
``curves.json`` (per-threshold curves + AUC/pAUC) and a ``summary.md`` table.

Each method is drawn as ONE polyline: the ROC only keeps thresholds whose FPR is
inside the x window (closed by an interpolated endpoint at the right border), and
the PR keeps the same thresholds so both panels share one operating range and no
curve plunges to precision ~0 at recall 1.

Usage:
    python tools/plot_roc_pr.py --manifest tmp/roc/manifest_v1.json \
        --outdir Ours/roc_pr/v2_20261105 --title "Seal test (1,983 images)"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Ours first (red); baselines get clearly separated hues, no near-duplicates.
COLORS = [
    '#d62728',  # Ours
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
    '#17becf', '#bcbd22', '#7f7f7f', '#393b79', '#637939', '#8c6d31',
]


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def curves(rec):
    """Per-threshold ROC/PR points from one histogram record.

    The returned plotting arrays are ordered the way the curves are read: from
    "predict nothing" (FPR = 0, TPR = 0) through decreasing thresholds to
    "predict everything" (1, 1). The (0, 0) start is a real operating point
    (threshold above the 8-bit maximum); without it every curve would begin at
    the first threshold that predicts anything and look truncated.
    """
    pos = np.asarray(rec['pos_hist'], dtype=np.float64)
    neg = np.asarray(rec['neg_hist'], dtype=np.float64)
    n_pos, n_neg = pos.sum(), neg.sum()
    tp = np.cumsum(pos[::-1])[::-1]              # pred >= t, indexed by t = 0..255
    fp = np.cumsum(neg[::-1])[::-1]
    t = np.arange(len(pos)).astype(np.float64)
    tpr = tp / max(n_pos, 1.0)
    fpr = fp / max(n_neg, 1.0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)

    down = np.arange(len(pos) - 1, -1, -1)      # t = 255 ... 0 → ascending TPR/FPR
    fpr_plot = np.concatenate([[0.0], fpr[down]])
    tpr_plot = np.concatenate([[0.0], tpr[down]])

    # PR: same thresholds, but t = 0 ("everything is positive", precision ≈ the
    # background rate) is degenerate and is dropped; the (0, 1) point at the top
    # of the ranking is the standard convention (it adds no area to AP).
    pr_idx = np.arange(len(pos) - 1, 0, -1)     # t = 255 ... 1
    pr_recall = np.concatenate([[0.0], tpr[pr_idx]])
    pr_precision = np.concatenate([[1.0], prec[pr_idx]])

    return {
        'threshold': t, 'tpr': tpr, 'fpr': fpr, 'precision': prec,
        'tp': tp, 'fp': fp, 'n_pos_px': float(n_pos), 'n_neg_px': float(n_neg),
        'n_operating_points': int(((tp + fp) > 0).sum()),
        'fpr_plot': fpr_plot, 'tpr_plot': tpr_plot,
        'pr_recall': pr_recall, 'pr_precision': pr_precision,
    }


def _trapezoid(y, x):
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return float(fn(y, x))


def auc(fpr, tpr):
    order = np.argsort(fpr)
    return _trapezoid(tpr[order], fpr[order])


def pauc(fpr, tpr, xmax):
    """Partial AUC over FPR in [0, xmax], normalised by xmax.

    A method that predicts (almost) nothing scores ~0 here, unlike the full-range
    AUC where "predict nothing" trivially wins — the same trap as S/MAE on Seal.
    """
    order = np.argsort(fpr)
    f, t = fpr[order], tpr[order]
    keep = f <= xmax
    f_in, t_in = f[keep], t[keep]
    if f_in.size == 0:
        return 0.0
    t_end = float(np.interp(xmax, f, t)) if f[-1] > xmax else float(t[-1])
    f_out = np.concatenate([f_in, [xmax]])
    t_out = np.concatenate([t_in, [t_end]])
    return _trapezoid(t_out, f_out) / xmax


def tpr_at(fpr, tpr, x):
    order = np.argsort(fpr)
    return float(np.interp(x, fpr[order], tpr[order]))


def ap(rec, prec):
    return float(np.sum(np.diff(rec) * prec[1:])) if rec.size > 1 else 0.0


def roc_points(cv, xmax):
    """ROC polyline clipped to FPR <= xmax (curves that leave the window are cut)."""
    f, t = cv['fpr_plot'], cv['tpr_plot']
    keep = f <= xmax
    return f[keep], t[keep]


def pr_points(cv, style='v1'):
    """PR polyline.

    style='v1': the full curve as computed, i.e. (0,1) at the top of the ranking
    and the degenerate t=0 point at recall 1 (the "bow-tie" look).
    style='v2': from the first point that detects something, no sentinel, no tail.
    """
    if style == 'v1':
        return cv['pr_recall'], cv['pr_precision']
    r, p = cv['pr_recall'], cv['pr_precision']
    keep = r > 0
    return r[keep], p[keep]


def at_threshold(cv, thr):
    i = int(np.argmin(np.abs(cv['threshold'] - thr)))
    return {'threshold': float(cv['threshold'][i]), 'tpr': float(cv['tpr'][i]),
            'fpr': float(cv['fpr'][i]),
            'precision': float(cv['tp'][i] / max(cv['tp'][i] + cv['fp'][i], 1.0))}


def style_axes(ax):
    ax.grid(True, linestyle=':', linewidth=0.55, color='0.72', alpha=0.9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, width=0.8, length=3)


def two_segment(cv, xmax, knee_frac):
    """Reduce a ROC curve to two straight segments: rise to the knee, then flat.

    The knee is the first point (ascending FPR) whose TPR reaches ``knee_frac`` of
    the TPR at the right border; the plateau is that border TPR. This is the
    "straight before the knee, straight after the knee" look of the IRSTD ROC
    figures, and it keeps every curve inside the window.
    """
    f, t = cv['fpr_plot'], cv['tpr_plot']
    plateau = float(np.interp(xmax, f, t)) if f[-1] > xmax else float(t[-1])
    keep = f <= xmax
    f_in, t_in = f[keep], t[keep]
    hit = np.flatnonzero(t_in >= knee_frac * plateau)
    knee_f = float(f_in[hit[0]]) if hit.size else 0.0
    knee_t = float(t_in[hit[0]]) if hit.size else 0.0
    return np.array([0.0, knee_f, xmax]), np.array([0.0, knee_t, plateau])


def draw_roc(ax, data, xmax, lw=1.15, lw_main=2.0, two=False, knee_frac=0.98):
    """Plot one polyline per method; the axes clip it at the window border."""
    for i, (label, cv) in enumerate(data):
        if two:
            f, t = two_segment(cv, xmax, knee_frac)
        else:
            f, t = cv['fpr_plot'], cv['tpr_plot']
        ax.plot(f, t, color=COLORS[i % len(COLORS)],
                lw=lw_main if i == 0 else lw,
                zorder=5 if i == 0 else 3,
                solid_capstyle='round', label=label)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.set_xticks(np.linspace(0, xmax, 5))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, -4), useMathText=True)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    style_axes(ax)


def draw_pr(ax, data, lw=1.15, lw_main=2.0, style='v1'):
    for i, (label, cv) in enumerate(data):
        r, p = pr_points(cv, style)
        ax.plot(r, p, color=COLORS[i % len(COLORS)],
                lw=lw_main if i == 0 else lw,
                zorder=5 if i == 0 else 3,
                solid_capstyle='round', label=label)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    style_axes(ax)


def save(fig, outdir, stem, dpi):
    png = os.path.join(outdir, f'{stem}.png')
    pdf = os.path.join(outdir, f'{stem}.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return png, pdf


def legend_outside(fig, ax, handles, labels, ncol=1, fontsize=7.4, gap=0.012):
    """Legend hugging the right edge of `ax` (figure coordinates).

    Anchoring at a fixed figure fraction left a wide blank band between the
    axes and the legend, which is what the caption/figure gap complaint was.
    """
    pos = ax.get_position()
    fig.legend(handles, labels, fontsize=fontsize, frameon=True, ncol=ncol,
               loc='center left',
               bbox_to_anchor=(pos.x1 + gap, pos.y0 + pos.height / 2.0),
               labelspacing=0.34, handlelength=2.1, borderpad=0.5,
               edgecolor='0.8')


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--manifest', required=True)
    ap_.add_argument('--outdir', required=True)
    ap_.add_argument('--title', default='')
    ap_.add_argument('--xmax', type=float, default=5e-4, help='FPR window for ROC and pAUC')
    ap_.add_argument('--dpi', type=int, default=600)
    ap_.add_argument('--pr-style', choices=('v1', 'v2'), default='v1',
                     help='v1 = full PR curve with the (0,1) sentinel; v2 = trimmed')
    ap_.add_argument('--fig-width', type=float, default=4.5,
                     help='single-panel figure width in inches (raise it for a longer x axis)')
    ap_.add_argument('--two-segment', action='store_true',
                     help='draw each ROC as straight rise + straight plateau (knee style)')
    ap_.add_argument('--knee-frac', type=float, default=0.98,
                     help='knee = first point whose TPR >= this fraction of the border TPR')
    args = ap_.parse_args()

    manifest = load(args.manifest)
    methods = manifest['methods'] if isinstance(manifest, dict) else manifest
    os.makedirs(args.outdir, exist_ok=True)

    data, table = [], []
    for m in methods:
        rec = load(m['hist'])
        cv = curves(rec)
        data.append((m['label'], cv))
        op = at_threshold(cv, 128)
        row = {
            'label': m['label'],
            'images': rec['images'],
            'auc_roc': auc(cv['fpr'], cv['tpr']),
            'pauc': pauc(cv['fpr_plot'], cv['tpr_plot'], args.xmax),
            'pauc_5e4': pauc(cv['fpr_plot'], cv['tpr_plot'], 5e-4),
            'ap_pr': ap(cv['pr_recall'], cv['pr_precision']),
            'fg_rate': cv['n_pos_px'] / (cv['n_pos_px'] + cv['n_neg_px']),
            'pd_at_128': op['tpr'],
            'fa_at_128': op['fpr'],
            'precision_at_128': op['precision'],
            'tpr_at_fpr': {f'{x:.1e}': tpr_at(cv['fpr_plot'], cv['tpr_plot'], x)
                           for x in (1e-4, 2e-4, args.xmax)},
            'n_operating_points': cv['n_operating_points'],
        }
        table.append(row)
        print(f'{m["label"]:12s} pAUC={row["pauc"]:.4f} AP={row["ap_pr"]:.4f} '
              f'TPR@5e-4={row["tpr_at_fpr"][f"{args.xmax:.1e}"]:.4f} '
              f'Pd@128={row["pd_at_128"]:.4f} Fa@128={row["fa_at_128"]:.3e} '
              f'AUC={row["auc_roc"]:.4f}')

    # ---- ROC only, one polyline per method (the reference figure's layout)
    fig, ax = plt.subplots(figsize=(args.fig_width, 3.3))
    draw_roc(ax, data, args.xmax, two=args.two_segment, knee_frac=args.knee_frac)
    handles, labels = ax.get_legend_handles_labels()
    legend_outside(fig, ax, handles, labels)
    save(fig, args.outdir, 'roc', args.dpi)

    # ---- PR only
    fig, ax = plt.subplots(figsize=(args.fig_width, 3.3))
    draw_pr(ax, data, style=args.pr_style)
    handles, labels = ax.get_legend_handles_labels()
    legend_outside(fig, ax, handles, labels)
    save(fig, args.outdir, 'pr', args.dpi)

    # ---- two panels sharing one legend
    fig, axes = plt.subplots(1, 2, figsize=(min(args.fig_width * 2 + 0.6, 11.5), 3.4))
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.9)
    draw_roc(axes[0], data, args.xmax, two=args.two_segment, knee_frac=args.knee_frac)
    axes[0].set_title('(a) ROC', fontsize=9.5)
    draw_pr(axes[1], data, style=args.pr_style)
    axes[1].set_title('(b) Precision-Recall', fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_outside(fig, axes[1], handles, labels)
    save(fig, args.outdir, 'roc_pr', args.dpi)

    # ---- numbers
    out = {
        'title': args.title,
        'xlim_fpr': args.xmax,
        'gt_positive_rule': 'gt > 127',
        'pred_threshold_rule': 'pred >= t (8-bit value)',
        'images': table[0]['images'] if table else 0,
        'methods': table,
        'curves': {label: {
            'fpr': cv['fpr'].tolist(), 'tpr': cv['tpr'].tolist(),
            'precision': cv['precision'].tolist(),
            'threshold': cv['threshold'].tolist(),
            'n_pos_px': cv['n_pos_px'], 'n_neg_px': cv['n_neg_px'],
            'n_operating_points': cv['n_operating_points'],
        } for label, cv in data},
    }
    with open(os.path.join(args.outdir, 'curves.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)

    key = f'{args.xmax:.1e}'
    lines = [
        f'# {args.title or "Seal test"}：ROC / PR 曲线数据',
        '',
        '像素级统计，GT 前景 `gt > 127`，预测阈值 `pred >= t`（8 位灰度值），'
        f'{table[0]["images"]} 张测试图全数据集累计。每个方法一条折线：'
        f'ROC 只画 FPR ≤ {key} 的部分（曲线出了窗口就截断），'
        'PR 画全部阈值（`t = 0` 的"全预测为前景"退化点除外）。',
        '',
        f'| 方法 | pAUC@{key} | pAUC@5.0e-04 | TPR@FPR={key} ↑ | Pd@t=128 ↑ | Fa@t=128 ↓ | Precision@t=128 | AUC(全范围) | AP(PR) ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in table:
        lines.append(
            f'| {r["label"]} | {r["pauc"]:.5f} | {r["pauc_5e4"]:.5f} | {r["tpr_at_fpr"][key]:.5f} | '
            f'{r["pd_at_128"]:.5f} | {r["fa_at_128"]:.3e} | '
            f'{r["precision_at_128"]:.5f} | {r["auc_roc"]:.5f} | {r["ap_pr"]:.5f} |')
    lines += [
        '',
        f'- **pAUC@{key}** = 图中窗口 [0, {key}] 的归一化面积（画多宽就积多宽）。',
        '- **pAUC@5.0e-04** = 各方法操作点（Fa@t=128 ≤ 2.4e-04）所在的低误检窗口，作为排名主指标。',
        '- **AUC(全范围)** 为 FPR ∈ [0, 1] 的全积分，只作参考：几乎不预测前景的方法'
        '（UIU-Net / SCTransNet / ILNet）在 Seal 上反而拿到 0.977 以上，'
        '与"S/MAE 被背景主导"是同一类陷阱，不能当排名依据。',
        '- **AP(PR)** 为按 recall 步进积分的 average precision（与 sklearn 同口径：recall=0 处补 (0,1)，'
        't=0 的"全预测为前景"退化点不参与）。',
        '- **Pd@t=128 / Fa@t=128** 取二值图阈值 `pred >= 128`，与八项指标同阈值，'
        '但这里仍是像素级 TPR/FPR，不是目标级 Pd/Fa。',
        '',
        '图：`roc.png/pdf`（参考图版式）、`pr.png/pdf`、`roc_pr.png/pdf`（并排两联）。',
    ]
    with open(os.path.join(args.outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {args.outdir}')


if __name__ == '__main__':
    main()
