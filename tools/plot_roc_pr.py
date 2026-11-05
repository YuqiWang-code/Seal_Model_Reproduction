#!/usr/bin/env python3
"""Draw pixel-level ROC / PR curves from the histograms written by tools/roc_hist.py.

One method = one ``<name>.json`` with the 256-bin foreground/background prediction
histograms. Every figure is saved twice (600 dpi PNG + vector PDF) together with a
``curves.json`` (per-threshold curves + AUC) and a ``summary.md`` table.

Usage:
    python tools/plot_roc_pr.py --manifest tmp/roc/manifest_v1.json \
        --outdir Ours/roc_pr/v1_20261105 --title "Seal test (1,983 images)"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Ours first (red), then a stable qualitative palette for the baselines.
COLORS = [
    '#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#98df8a', '#ffbb78', '#c5b0d5',
    '#f7b6d2', '#c49c94', '#dbdb8d', '#9edae5',
]


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def curves(rec):
    """Return dict of per-threshold curves from one histogram record."""
    pos = np.asarray(rec['pos_hist'], dtype=np.float64)
    neg = np.asarray(rec['neg_hist'], dtype=np.float64)
    n_pos, n_neg = pos.sum(), neg.sum()
    t = np.arange(len(pos))                      # t = prediction value threshold
    tp = np.concatenate([np.cumsum(pos[::-1])[::-1], [0.0]])   # pred >= t, t = 0..255, plus 256
    fp = np.concatenate([np.cumsum(neg[::-1])[::-1], [0.0]])
    tpr = tp / max(n_pos, 1.0)
    fpr = fp / max(n_neg, 1.0)

    # PR only has a defined precision where at least one pixel is predicted
    # positive; the standard convention is a (recall 0, precision 1) sentinel.
    valid = (tp + fp) > 0
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=valid)
    rec_valid = tpr[valid]
    prec_valid = prec[valid]
    order = np.argsort(rec_valid, kind='stable')
    rec_valid, prec_valid = rec_valid[order], prec_valid[order]
    rec_valid = np.concatenate([[0.0], rec_valid])
    prec_valid = np.concatenate([[1.0], prec_valid])

    return {
        'threshold': np.concatenate([t, [256]])[valid],
        'tpr': tpr, 'fpr': fpr,
        'pr_threshold': np.concatenate([t, [256]])[valid],
        'pr_recall': rec_valid, 'pr_precision': prec_valid,
        'tp': tp, 'fp': fp,
        'n_pos_px': float(n_pos), 'n_neg_px': float(n_neg),
        'n_operating_points': int(valid.sum()),
    }


def _trapezoid(y, x):
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return float(fn(y, x))


def auc(fpr, tpr):
    """Area under the ROC curve over the full FPR range (trapezoid on sorted FPR)."""
    order = np.argsort(fpr)
    f, t = fpr[order], tpr[order]
    return _trapezoid(t, f)


def pauc(fpr, tpr, xmax):
    """Partial AUC restricted to FPR in [0, xmax], normalised by xmax.

    A method that predicts (almost) nothing scores ~0 here, unlike the full-range
    AUC where "predict nothing" trivially wins — the same trap as S/MAE on Seal.
    """
    order = np.argsort(fpr)
    f, t = fpr[order], tpr[order]
    keep = f <= xmax
    f, t = f[keep], t[keep]
    # interpolated TPR exactly at xmax
    t_end = float(np.interp(xmax, f, t)) if f.size > 1 else (float(t[-1]) if t.size else 0.0)
    f = np.concatenate([f, [xmax]])
    t = np.concatenate([t, [t_end]])
    return _trapezoid(t, f) / xmax


def tpr_at(fpr, tpr, x):
    order = np.argsort(fpr)
    return float(np.interp(x, fpr[order], tpr[order]))


def ap(rec, prec):
    """Average precision: step-wise integral of precision over recall."""
    return float(np.sum(np.diff(rec) * prec[1:])) if rec.size > 1 else 0.0


def at_threshold(cv, thr):
    i = int(np.argmin(np.abs(cv['threshold'] - thr)))
    return {'threshold': float(cv['threshold'][i]), 'tpr': float(cv['tpr'][i]),
            'fpr': float(cv['fpr'][i]),
            'precision': float(cv['tp'][i] / max(cv['tp'][i] + cv['fp'][i], 1.0))}


def style_axes(ax):
    ax.grid(True, linestyle=':', linewidth=0.6, color='0.7', alpha=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, width=0.8)


def plot_roc(ax, data, xmax, legend=False, legend_out=True):
    for i, (label, cv) in enumerate(data):
        ax.plot(cv['fpr'], cv['tpr'], lw=1.7 if i == 0 else 1.1,
                color=COLORS[i % len(COLORS)], label=label,
                zorder=5 if i == 0 else 3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, -4), useMathText=True)
    style_axes(ax)
    if legend:
        if legend_out:
            ax.legend(fontsize=6.6, frameon=False, loc='center left',
                      bbox_to_anchor=(1.01, 0.5), labelspacing=0.32, handlelength=1.9)
        else:
            ax.legend(fontsize=6.6, frameon=False, loc='lower right', labelspacing=0.32)


def plot_pr(ax, data, legend=False, legend_out=True):
    for i, (label, cv) in enumerate(data):
        ax.plot(cv['pr_recall'], cv['pr_precision'], lw=1.7 if i == 0 else 1.1,
                color=COLORS[i % len(COLORS)], label=label,
                zorder=5 if i == 0 else 3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    style_axes(ax)
    if legend:
        if legend_out:
            ax.legend(fontsize=6.6, frameon=False, loc='center left',
                      bbox_to_anchor=(1.01, 0.5), labelspacing=0.32, handlelength=1.9)
        else:
            ax.legend(fontsize=6.6, frameon=False, loc='lower left', labelspacing=0.32)


def save(fig, outdir, stem, dpi):
    png = os.path.join(outdir, f'{stem}.png')
    pdf = os.path.join(outdir, f'{stem}.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return png, pdf


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--manifest', required=True)
    ap_.add_argument('--outdir', required=True)
    ap_.add_argument('--title', default='Seal test set')
    ap_.add_argument('--xmax', type=float, default=5e-4)
    ap_.add_argument('--dpi', type=int, default=600)
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
            'pauc': pauc(cv['fpr'], cv['tpr'], args.xmax),
            'ap_pr': ap(cv['pr_recall'], cv['pr_precision']),
            'fg_rate': cv['n_pos_px'] / (cv['n_pos_px'] + cv['n_neg_px']),
            'pd_at_128': op['tpr'],
            'fa_at_128': op['fpr'],
            'precision_at_128': op['precision'],
            'tpr_at_fpr': {f'{x:.1e}': tpr_at(cv['fpr'], cv['tpr'], x)
                           for x in (1e-4, 2e-4, args.xmax)},
            'n_operating_points': cv['n_operating_points'],
        }
        table.append(row)
        print(f'{m["label"]:12s} AUC={row["auc_roc"]:.4f} pAUC={row["pauc"]:.4f} '
              f'AP={row["ap_pr"]:.4f} Pd@128={row["pd_at_128"]:.4f} '
              f'Fa@128={row["fa_at_128"]:.3e} '
              f'TPR@5e-4={row["tpr_at_fpr"][f"{args.xmax:.1e}"]:.4f}')

    # ---- single-panel ROC
    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    plot_roc(ax, data, args.xmax, legend=True)
    save(fig, args.outdir, 'roc', args.dpi)

    # ---- single-panel PR
    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    plot_pr(ax, data, legend=True)
    save(fig, args.outdir, 'pr', args.dpi)

    # ---- combined (a) ROC (b) PR, same look as the reference figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    plot_roc(axes[0], data, args.xmax)
    axes[0].set_title('(a) ROC', fontsize=9.5)
    plot_pr(axes[1], data)
    axes[1].set_title('(b) Precision-Recall', fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.2, frameon=False, ncol=1,
               loc='center left', bbox_to_anchor=(0.995, 0.5), labelspacing=0.35)
    fig.suptitle(args.title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.94))
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
            'pr_recall': cv['pr_recall'].tolist(),
            'pr_precision': cv['pr_precision'].tolist(),
            'pr_threshold': cv['pr_threshold'].tolist(),
            'n_pos_px': cv['n_pos_px'], 'n_neg_px': cv['n_neg_px'],
            'n_operating_points': cv['n_operating_points'],
        } for label, cv in data},
    }
    with open(os.path.join(args.outdir, 'curves.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)

    key = f'{args.xmax:.1e}'
    lines = [
        f'# {args.title}：ROC / PR 曲线数据',
        '',
        '像素级统计，GT 前景 `gt > 127`，预测阈值 `pred >= t`（8 位灰度值），'
        f'{table[0]["images"]} 张测试图全数据集累计。8 位预测图最多给出 256 个操作点，'
        '曲线为这些点直接连线（未做平滑）。',
        '',
        f'| 方法 | pAUC | TPR@FPR={key} ↑ | Pd@t=128 ↑ | Fa@t=128 ↓ | Precision@t=128 | AUC(全范围) | AP(PR) ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in table:
        lines.append(
            f'| {r["label"]} | {r["pauc"]:.5f} | {r["tpr_at_fpr"][key]:.5f} | '
            f'{r["pd_at_128"]:.5f} | {r["fa_at_128"]:.3e} | '
            f'{r["precision_at_128"]:.5f} | {r["auc_roc"]:.5f} | {r["ap_pr"]:.5f} |')
    lines += [
        '',
        f'- **pAUC** = FPR ∈ [0, {key}] 内的梯形面积再除以 {key}，即图中可见部分的归一化面积；'
        '这是本图的主指标。',
        '- **AUC(全范围)** 为 FPR ∈ [0, 1] 的全范围积分，只作参考：几乎不预测前景的方法'
        '（如 UIU-Net、SCTransNet、ILNet）在 Seal 上反而拿到 0.97 以上的全范围 AUC，'
        '与"S/MAE 被背景主导"是同一类陷阱，不能当排名依据。',
        '- **AP(PR)** 为按 recall 步进积分的 average precision（PR 曲线在 recall=0 处按惯例补 (0, 1) 哨兵点）。',
        '- **Pd@t=128 / Fa@t=128** 取二值图阈值 `pred >= 128`，与八项指标同阈值，'
        '但这里仍是像素级 TPR/FPR，不是目标级 Pd/Fa。',
        '',
        '图：`roc.png/pdf`（单幅 ROC）、`pr.png/pdf`（单幅 P-R）、`roc_pr.png/pdf`（并排两联图）。',
    ]
    with open(os.path.join(args.outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {args.outdir}')


if __name__ == '__main__':
    main()
