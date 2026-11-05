#!/usr/bin/env python3
"""Draw target-level Pd vs Fa curves (the "ROC" style used in IRSTD papers).

Input: the JSONs written by tools/target_curves.py (Pd / Fa at a threshold grid).
Each method is one polyline: (0, 0) → points sorted by Fa → clipped at the right
border of the Fa window. Saves PNG (600 dpi) + PDF + curves.json + summary.md.

Usage:
    python tools/plot_target_curves.py --manifest tmp/roc/manifest_target_v1.json \
        --outdir Ours/roc_pr/v3_20261105 --title "Seal test (1,983 images)"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    '#d62728',  # Ours
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
    '#17becf', '#bcbd22', '#7f7f7f', '#393b79', '#637939', '#8c6d31',
]


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def points(rec, xmax):
    fa = np.asarray(rec['Fa'], dtype=float)
    pd = np.asarray(rec['Pd'], dtype=float)
    order = np.argsort(fa)
    fa, pd = fa[order], pd[order]
    keep = fa <= xmax
    fa, pd = fa[keep], pd[keep]
    fa = np.concatenate([[0.0], fa])
    pd = np.concatenate([[0.0], pd])
    return fa, pd


def auc_pd_fa(fa, pd, xmax):
    """Normalised area under the Pd-Fa curve over Fa in [0, xmax]."""
    keep = fa <= xmax
    f, p = fa[keep], pd[keep]
    if f.size == 0:
        return 0.0
    f = np.concatenate([f, [xmax]])
    p = np.concatenate([p, [p[-1]]])
    return float(np.trapezoid(p, f) / xmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--title', default='Seal test set')
    ap.add_argument('--xmax', type=float, default=5e-4)
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    manifest = load(args.manifest)
    methods = manifest['methods'] if isinstance(manifest, dict) else manifest
    os.makedirs(args.outdir, exist_ok=True)

    data, table = [], []
    for m in methods:
        rec = load(m['json'])
        fa, pd = points(rec, args.xmax)
        data.append((m['label'], fa, pd, rec))
        idx128 = int(np.argmin(np.abs(np.asarray(rec['thresholds']) - 128)))
        table.append({
            'label': m['label'],
            'images': rec['images'],
            'n_gt_targets': rec['n_gt_targets'],
            'auc_pd_fa': auc_pd_fa(fa, pd, args.xmax),
            'pd_at_128': float(rec['Pd'][idx128]),
            'fa_at_128': float(rec['Fa'][idx128]),
            'max_pd': float(np.max(rec['Pd'])),
            'min_fa_any': float(min(f for f in rec['Fa'] if f > 0)) if any(f > 0 for f in rec['Fa']) else 0.0,
        })
        print(f'{m["label"]:12s} AUC(Pd-Fa)={table[-1]["auc_pd_fa"]:.4f} '
              f'maxPd={table[-1]["max_pd"]:.4f} Pd@128={table[-1]["pd_at_128"]:.4f} '
              f'Fa@128={table[-1]["fa_at_128"]:.3e}')

    def draw(ax, lw=1.15, lw_main=2.0):
        for i, (label, fa, pd, _) in enumerate(data):
            ax.plot(fa, pd, color=COLORS[i % len(COLORS)],
                    lw=lw_main if i == 0 else lw,
                    zorder=5 if i == 0 else 3,
                    solid_capstyle='round', label=label)
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('False Alarm Rate (Fa)', fontsize=9)
        ax.set_ylabel('Probability of Detection (Pd)', fontsize=9)
        ax.set_xticks(np.linspace(0, args.xmax, 6))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, -4), useMathText=True)
        ax.grid(True, linestyle=':', linewidth=0.55, color='0.72', alpha=0.9)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, width=0.8, length=3)

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    draw(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.4, frameon=True, loc='center left',
               bbox_to_anchor=(1.02, 0.5), labelspacing=0.34, handlelength=2.1,
               borderpad=0.5, edgecolor='0.8')
    fig.tight_layout(rect=(0, 0, 0.74, 1))
    fig.savefig(os.path.join(args.outdir, 'pd_fa.png'), dpi=args.dpi,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(args.outdir, 'pd_fa.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    out = {
        'title': args.title, 'xlim_fa': args.xmax,
        'match_rule': '8-connectivity, GT centroid distance < 3 px, one-to-one, raster order',
        'gt_positive_rule': 'gt > 127',
        'methods': table,
        'curves': {label: {'fa': fa.tolist(), 'pd': pd.tolist()}
                   for label, fa, pd, _ in data},
    }
    with open(os.path.join(args.outdir, 'pd_fa_curves.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)

    key = f'{args.xmax:.1e}'
    lines = [
        f'# {args.title}：目标级 Pd–Fa 曲线数据',
        '',
        '目标级统计（与八项指标的 Pd/Fa 同一定义）：8 连通、GT 质心距离 < 3 px、'
        '按 raster 顺序一对一匹配；Fa 为未匹配预测连通域面积 / 总像素数，'
        f'每个方法在 {table[0]["images"]} 张测试图（{table[0]["n_gt_targets"]} 个 GT 目标）上累计。',
        '',
        f'| 方法 | AUC(Pd-Fa, Fa≤{key}) ↑ | max Pd ↑ | Pd@t=128 ↑ | Fa@t=128 ↓ |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in table:
        lines.append(f'| {r["label"]} | {r["auc_pd_fa"]:.5f} | {r["max_pd"]:.5f} | '
                     f'{r["pd_at_128"]:.5f} | {r["fa_at_128"]:.3e} |')
    lines += [
        '',
        '阈值网格为 255→200 步 2、200→120 步 4、120→40 步 8、40→1 步 8，共 63 个点；'
        '曲线按 Fa 升序连线，起点补 (0, 0)（阈值高于 8 位最大值 = 不预测任何前景）。',
        'Pd@t=128 / Fa@t=128 应与八项指标表逐位一致（同一定义、同一批预测）。',
        '',
        '图：`pd_fa.png/pdf`。',
    ]
    with open(os.path.join(args.outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {args.outdir}')


if __name__ == '__main__':
    main()
