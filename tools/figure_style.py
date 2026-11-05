#!/usr/bin/env python3
"""One font/label/format spec shared by every curve figure of this project.

`roc.pdf` and `pr_obj.pdf` sit side by side in the paper, so their ticks, axis
labels, grid and legend must be typographically identical. Everything that used to
be hard-coded twice now lives here:

    axis label    9.0 pt  x --font-scale
    tick labels   8.0 pt  x --font-scale
    legend        6.4 pt  x --font-scale
    tick marks    0.8 pt wide, 3 pt long, pointing out, 2.5 pt padding
    y tick text   '%.1f'  (0.0, 0.2, ... 1.0)
    grid          dotted, 0.55 pt, 0.72 grey
    spines        top/right hidden, 0.8 pt

Usage from a plotter::

    from figure_style import apply, axis_labels, finish, LEGEND_PT
    apply(fs)
    ...
    axis_labels(ax, 'Recall', 'Precision', fs)
    finish(ax, fs)
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

LABEL_PT = 9.0
TICK_PT = 8.0
LEGEND_PT = 6.4
GRID = dict(linestyle=':', linewidth=0.55, color='0.72', alpha=0.9)


def apply(font_scale=1.0):
    """Global rcParams: same family everywhere, ticks sized by the scale."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': TICK_PT * font_scale,
        'mathtext.fontset': 'dejavusans',
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


def axis_labels(ax, xlabel, ylabel, font_scale=1.0):
    ax.set_xlabel(xlabel, fontsize=LABEL_PT * font_scale)
    ax.set_ylabel(ylabel, fontsize=LABEL_PT * font_scale)


def finish(ax, font_scale=1.0, y_ticks=None):
    """Grid, spines, tick marks and the shared one-decimal y tick format."""
    ax.grid(True, **GRID)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_PT * font_scale, width=0.8, length=3,
                   direction='out', pad=2.5)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    return ax
