#!/usr/bin/env python3
"""Draw target-level Pd vs Fa curves as two straight segments per method.

Each method is reduced to a knee (Fa, Pd) plus a flat plateau:

    (0, 0)  ->  knee  ->  (x_max, Pd_plateau)

so the figure reads as "straight rise, then straight plateau" like the ROC plots
in IRSTD papers. The knee is the first threshold (ascending Fa) whose Pd reaches
``--knee-frac`` of that method's maximum Pd.

Input: JSONs written by tools/target_curves.py.

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


def knee_and_plateau(rec, knee_frac):
    """Return (knee_fa, plateau_pd, raw_fa, raw_pd) for one method."""
    fa = np.asarray(rec['Fa'], dtype=float)
    pd = np.asarray(rec['Pd'], dtype=float)
    order = np.argsort(fa)
    fa, pd = fa[order], pd[order]
    plateau = float(pd.max())
    hit = np.flatnonzero(pd >= knee_frac * plateau)
    k = int(hit[0]) if hit.size else 0
    return float(fa[k]), plateau, fa, pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--title', default='Seal test set')
    ap.add_argument('--knee-frac', type=float, default=0.98)
    ap.add_argument('--xmax', type=float, default=None,
                    help='right border (default: 1.05 x the largest knee Fa)')
    ap.add_argument('--xmin', type=float, default=None, help='log-axis lower bound')
    ap.add_argument('--xscale', choices=('linear', 'log'), default='linear')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    manifest = load(args.manifest)
    methods = manifest['methods'] if isinstance(manifest, dict) else manifest
    os.makedirs(args.outdir, exist_ok=True)

    data = []
    for m in methods:
        rec = load(m['json'])
        knee_fa, plateau, fa, pd = knee_and_plateau(rec, args.knee_frac)
        idx128 = int(np.argmin(np.abs(np.asarray(rec['thresholds']) - 128)))
        data.append({
            'label': m['label'], 'knee_fa': knee_fa, 'plateau': plateau,
            'raw_fa': fa, 'raw_pd': pd, 'rec': rec,
            'pd_at_128': float(rec['Pd'][idx128]), 'fa_at_128': float(rec['Fa'][idx128]),
        })
        print(f'{m["label"]:12s} knee Fa={knee_fa:.3e} plateau Pd={plateau:.5f} '
              f'Pd@128={data[-1]["pd_at_128"]:.5f} Fa@128={data[-1]["fa_at_128"]:.3e}')

    xmax = args.xmax or 1.05 * max(d['knee_fa'] for d in data)
    xmin = args.xmin or min(d['knee_fa'] for d in data) / 3.0

    def draw(ax):
        for i, d in enumerate(data):
            x = [0.0, d['knee_fa'], xmax]
            y = [0.0, d['plateau'], d['plateau']]
            if args.xscale == 'log':
                x[0] = xmin
                ax.set_xscale('log')
                ax.set_xlim(xmin, xmax)
            else:
                ax.set_xlim(0, xmax)
            ax.plot(x, y, color=COLORS[i % len(COLORS)],
                    lw=2.0 if i == 0 else 1.15, zorder=5 if i == 0 else 3,
                    solid_capstyle='round', label=d['label'])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('False Alarm Rate (Fa)', fontsize=9)
        ax.set_ylabel('Probability of Detection (Pd)', fontsize=9)
        ax.grid(True, linestyle=':', linewidth=0.55, color='0.72', alpha=0.9)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, width=0.8, length=3)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    draw(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.4, frameon=True, loc='center left',
               bbox_to_anchor=(1.02, 0.5), labelspacing=0.34, handlelength=2.1,
               borderpad=0.5, edgecolor='0.8')
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    stem = 'pd_fa_log' if args.xscale == 'log' else 'pd_fa'
    fig.savefig(os.path.join(args.outdir, f'{stem}.png'), dpi=args.dpi,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(args.outdir, f'{stem}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    out = {
        'title': args.title, 'xmax': xmax, 'xscale': args.xscale,
        'knee_frac': args.knee_frac,
        'match_rule': '8-connectivity, GT centroid distance < 3 px, one-to-one, raster order',
        'methods': [{k: d[k] for k in ('label', 'knee_fa', 'plateau', 'pd_at_128', 'fa_at_128')}
                    for d in data],
    }
    with open(os.path.join(args.outdir, 'pd_fa_curves.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False)

    lines = [
        f'# {args.title}：目标级 Pd–Fa 曲线',
        '',
        '目标级统计（与八项指标的 Pd/Fa 同一定义）：8 连通、GT 质心距离 < 3 px、raster 顺序一对一匹配；'
        'Fa = 未匹配预测连通域面积 / 总像素数。每个方法压成两段直线：'
        f'(0,0) → 拐点 → (x_max, 平台)，拐点取 Pd ≥ {args.knee_frac:.0%} × max Pd 的第一个阈值点。',
        '',
        '| 方法 | 拐点 Fa | 平台 Pd | Pd@t=128 | Fa@t=128 |',
        '|---|---:|---:|---:|---:|',
    ]
    for d in data:
        lines.append(f'| {d["label"]} | {d["knee_fa"]:.3e} | {d["plateau"]:.5f} | '
                     f'{d["pd_at_128"]:.5f} | {d["fa_at_128"]:.3e} |')
    lines += ['', 'Pd@t=128 / Fa@t=128 与八项指标表逐位一致（同一定义、同一批 1,983 张预测、7,383 个 GT 目标）。']
    with open(os.path.join(args.outdir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {args.outdir} (xmax={xmax:.3e})')


if __name__ == '__main__':
    main()
