#!/usr/bin/env python3
"""Build the PDDiff-style qualitative figure for Seal.

Layout mirrors PDDiff Figs. 8/9: one header row, then for every sample two rows

    header :  Image | Mask | method_1 | ... | Ours
    row 1  :  input image | binary GT | binary predictions (red ROI box + zoom inset)
    row 2  :  3D response surfaces of the same columns (matplotlib plot_surface, jet)

2D panels are composed with OpenCV (white foreground on black, red ROI box,
red-bordered zoom inset bottom-right); the header bars and the 3D row use
matplotlib so the output stays vector friendly.

Predictions are 8-bit PNGs: binary panels threshold at 128 (prob > 0.5, the
project rule), the 3D surfaces use the raw 8-bit values.

Example
-------
    python tools/build_paper_qualitative.py \
        --samples docs/Seal定性对比样例.json --num 2 --image T \
        --methods DNA-Net=…/results/DNANet/Seal/test_binary \
                  MSHNet=…/results/MSHNet/Seal/test_binary \
                  HDNet=…/results/HDNet/Seal/test_binary \
                  L2SKNet=…/results/L2SKNet/Seal/test_binary \
        --ours Ours=…/results/Ours/Seal/test_binary \
        --out …/qualitative_paper --tag seal
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

SEAL = Path('/share_datasets/Seal_Model_Reproduction')

HEADER_COLORS = {
    'image': '#ED7D31',
    'mask': '#F8CBAD',
    'baseline': '#9DC3E6',
    'ours': '#2E75B6',
}
RED = (0, 0, 255)          # BGR
ZOOM = 10
MARGIN = 32
MIN_BOX = 80

# 3D 响应面配色：底色深蓝（近乎黑），峰顶亮红——和压平的背景形成强对比
SURF_CMAP = LinearSegmentedColormap.from_list('seal3d', [
    (0.00, '#0a1746'),
    (0.22, '#1f5fbe'),
    (0.40, '#22c6d8'),
    (0.60, '#ffe14d'),
    (0.74, '#ff8a1f'),
    (0.86, '#ff2d2d'),
    (1.00, '#e00b2b'),
])


def load_image(stem, modality, gamma=1.0, stretch='percentile'):
    """Display copy of the input frame.

    Seal thermal frames are min-max stretched already, but a single hot target
    pins the maximum and leaves the background at ~0.2, which prints almost black.
    For the figure only, the background is re-stretched with a 0.5-99.5 percentile
    window (the target still clips to 255) and an optional gamma is applied.
    """
    if modality.upper() == 'RGB':
        img = cv2.imread(str(SEAL / 'Seal_data/test/RGB' / f'{stem}.jpg'), cv2.IMREAD_GRAYSCALE)
    else:
        raw = cv2.imread(str(SEAL / 'Seal_data/test/T' / f'{stem}.tif'), cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img is None:
        raise RuntimeError(f'cannot read image for {stem}')
    if stretch == 'percentile':
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi - lo > 1:
            img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if gamma != 1.0:
        lut = np.array([min(255, int(round(255 * (i / 255.0) ** gamma))) for i in range(256)], np.uint8)
        img = cv2.LUT(img, lut)
    return img


def load_map(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'cannot read {path}')
    return img


def roi_from_gt(gt, margin=MARGIN, min_box=MIN_BOX):
    ys, xs = np.where(gt > 127)
    h, w = gt.shape
    if len(ys) == 0:
        return 0, min(h, min_box), 0, min(w, min_box)
    cy, cx = int(ys.mean()), int(xs.mean())
    span = max(int(ys.max() - ys.min()), int(xs.max() - xs.min()))
    half = max(min_box // 2, span // 2 + margin)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return y0, y1, x0, x1


def compose_panel(sheet, roi, with_box=True, with_inset=True, zoom=ZOOM):
    panel = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    y0, y1, x0, x1 = roi
    if with_box:
        cv2.rectangle(panel, (x0, y0), (x1, y1), RED, 2, cv2.LINE_AA)
    if not with_inset:
        return panel
    crop = sheet[y0:y1, x0:x1]
    inset = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    ih, iw = inset.shape
    ph, pw = panel.shape[:2]
    max_w, max_h = int(pw * 0.46), int(ph * 0.46)
    scale = min(max_w / iw, max_h / ih, 1.0)
    if scale < 1.0:
        inset = cv2.resize(inset, (max(1, int(iw * scale)), max(1, int(ih * scale))),
                           interpolation=cv2.INTER_AREA)
    ih, iw = inset.shape
    ox, oy = pw - iw - 2, ph - ih - 2
    panel[oy:oy + ih, ox:ox + iw] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (ox - 2, oy - 2), (ox + iw + 1, oy + ih + 1), RED, 2, cv2.LINE_AA)
    return panel


def diag_panel(pred_bin, gt_bin, gt, roi, roi_mode, target=384, box_pad=8):
    """TP/FP/FN colour panel: TP white, FP red, FN green, plus a red GT box.

    Used for the Visio assembly so each method's error pattern (miss = green,
    false alarm = red) is visible at a glance.
    """
    h, w = gt_bin.shape
    colour = np.zeros((h, w, 3), np.uint8)
    colour[pred_bin & gt_bin] = (255, 255, 255)
    colour[pred_bin & ~gt_bin] = (0, 0, 255)      # BGR red
    colour[~pred_bin & gt_bin] = (0, 255, 0)      # BGR green

    ys, xs = np.where(gt_bin)
    if len(ys):
        gx0, gx1 = int(xs.min()), int(xs.max())
        gy0, gy1 = int(ys.min()), int(ys.max())
    else:
        gx0 = gx1 = gy0 = gy1 = 0

    y0, y1, x0, x1 = roi
    if roi_mode:
        crop = colour[y0:y1, x0:x1]
        side = max(crop.shape[:2])
        k = max(1, int(round(target / max(1, side))))
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        ch, cw = crop.shape[:2]
        canvas = np.zeros((max(ch, cw), max(ch, cw), 3), np.uint8)
        oy, ox = (canvas.shape[0] - ch) // 2, (canvas.shape[1] - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop
        # GT box in the upscaled ROI
        bx0 = ox + (gx0 - x0) * k - box_pad
        bx1 = ox + (gx1 - x0) * k + box_pad
        by0 = oy + (gy0 - y0) * k - box_pad
        by1 = oy + (gy1 - y0) * k + box_pad
        thick = 2
        cv2.rectangle(canvas, (max(0, bx0), max(0, by0)), (min(canvas.shape[1] - 1, bx1),
                      min(canvas.shape[0] - 1, by1)), RED, thick, cv2.LINE_AA)
        return canvas

    panel = colour.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), RED, 2, cv2.LINE_AA)
    return panel


def roi_panel(sheet, roi, target=384):
    """Panel = the ROI itself, upscaled with nearest-neighbour (no inset)."""
    y0, y1, x0, x1 = roi
    crop = sheet[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    k = max(1, int(round(target / max(1, side))))
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    h, w = crop.shape[:2]
    canvas = np.zeros((max(h, w), max(h, w)), np.uint8)
    oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def safe_name(name):
    return name.replace('+', 'p').replace('/', '_').replace(' ', '_')


def trim_white(img_bgr, pad=10, thr=250):
    """Crop the surrounding white border (keeps a small even pad)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g < thr)
    if len(ys) == 0:
        return img_bgr
    h, w = g.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    return img_bgr[y0:y1, x0:x1]


def render_surface_png(arr, path, vmax, cmap='jet',
                       size_in=(0.873, 0.733), dpi=1200, pad=6):
    """One 3D box as a standalone PNG, tight-cropped so the box fills the frame.

    `arr` is the already-smoothed response crop. Rendered at one cell of the composite
    figure (0.873 x 0.733 in) with the same tick font, then the white border is trimmed
    to `pad` px, so pasting it back with aspect='auto' fills the cell.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = plt.figure(figsize=size_in, dpi=dpi, facecolor='white')
    ax = f.add_subplot(111, projection='3d')
    add_surface(ax, arr, vmax=vmax, cmap=cmap)
    f.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    import io

    buf = io.BytesIO()
    f.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(f)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = trim_white(img, pad=pad)
    cv2.imwrite(str(path), img)
    return path


def add_surface(ax, arr, vmax=255.0, cmap=SURF_CMAP, tick_font=1.7):
    """PDDiff Fig. 9 style surface: filled frame, light grid, thin grey axes, tick labels.

    `arr` must already be smoothed; `vmax` is normally the peak of `arr`, so the target
    response always reaches the top of the box (tall, obvious peak) and gets the hot end
    of the colour map.
    """
    h, w = arr.shape
    step = max(1, min(h, w) // 48)
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    zz = arr[::step, ::step]
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
    ax.plot_surface(xx, yy, zz, cmap=cmap, norm=norm, rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)
    ax.set_zlim(0, vmax)
    ax.view_init(elev=26, azim=-64)

    # 坐标刻度：X/Y 是裁剪块像素坐标，Z 是响应值；取整数"好刻度"，与 PDDiff 一致
    loc = MaxNLocator(nbins=4, integer=True, steps=[1, 2, 2.5, 5, 10])
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis='x', labelsize=tick_font, pad=-3.0, length=0.4, width=0.15,
                   colors='0.45')
    ax.tick_params(axis='y', labelsize=tick_font, pad=-3.0, length=0.4, width=0.15,
                   colors='0.45')
    ax.tick_params(axis='z', labelsize=tick_font, pad=-1.5, length=0.4, width=0.15,
                   colors='0.45')
    ax.set_xlabel('X', fontsize=tick_font + 0.3, labelpad=-9, color='0.15')
    ax.set_ylabel('Y', fontsize=tick_font + 0.3, labelpad=-9, color='0.15')
    ax.set_zlabel('Z', fontsize=tick_font + 0.3, labelpad=-4, color='0.15')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1))   # 细浅灰外框
        axis.pane.set_linewidth(0.3)
        axis._axinfo['grid'].update(color=(0.86, 0.86, 0.86, 1), linewidth=0.22)
        # 去掉 matplotlib 默认的黑色粗轴线，改成很细的灰线（PDDiff 没有黑粗线）
        axis.line.set_color((0.55, 0.55, 0.55, 1))
        axis.line.set_linewidth(0.22)

    ax.grid(True)
    ax.set_box_aspect((1, 1, 1.00), zoom=0.88)
    ax.margins(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--num', type=int, default=2)
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--methods', nargs='+', default=[])
    ap.add_argument('--ours', default=None)
    ap.add_argument('--surf-maps', nargs='+', default=[],
                    help='name=dir of grey response maps used for the 3D row when the 2D row '
                         'needs a binarised map (e.g. name=results/X/Seal/fullsup_test_maps)')
    ap.add_argument('--image', default='T', choices=['T', 'RGB'])
    ap.add_argument('--image-gamma', type=float, default=1.0, help='display-only gamma for the Image column')
    ap.add_argument('--image-stretch', default='percentile', choices=['none', 'percentile'],
                    help='display-only background stretch for the Image column')
    ap.add_argument('--crop3d', type=int, default=192)
    ap.add_argument('--split-dir', default=None,
                    help='also write every panel as its own PNG under '
                         '<dir>/<nn>_<id>/2d/<col>.png and .../3d/<col>.png')
    ap.add_argument('--roi-min', type=int, default=MIN_BOX, help='minimum ROI side (px)')
    ap.add_argument('--roi-margin', type=int, default=MARGIN, help='padding around the GT bbox (px)')
    ap.add_argument('--diag', action='store_true',
                    help='2D panels = TP white / FP red / FN green + red GT box (for Visio)')
    ap.add_argument('--roi-panels', action='store_true',
                    help='2D row = upscaled ROI instead of full frame + inset')
    ap.add_argument('--z-mode', default='per-panel', choices=['shared', 'per-panel'])
    ap.add_argument('--out', default=str(SEAL / 'qualitative_paper'))
    ap.add_argument('--tag', default='seal')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--fig-width', type=float, default=7.2, help='figure width in inches')
    ap.add_argument('--header-font', type=float, default=8.0, help='header label font size')
    ap.add_argument('--font', default='Microsoft YaHei,Noto Sans CJK JP,DejaVu Sans',
                    help='comma separated font fallback list (first installed family wins)')
    ap.add_argument('--sample-labels', action='store_true', help='add (a)/(b) labels under each block')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split_root = Path(args.split_dir) if args.split_dir else None
    if split_root is not None:
        split_root.mkdir(parents=True, exist_ok=True)
    cache_root = out / '_3d_cache'
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.font:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [f.strip() for f in args.font.split(',') if f.strip()]
        plt.rcParams['axes.unicode_minus'] = False
    from matplotlib import font_manager
    # fontconfig-only installs (.ttc) are invisible to matplotlib until registered
    for cand in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
        if Path(cand).exists():
            try:
                font_manager.fontManager.addfont(cand)
            except Exception:  # noqa: BLE001
                pass
    resolved = font_manager.findfont(font_manager.FontProperties(family=plt.rcParams['font.sans-serif']))
    print('font ->', resolved)

    samples = json.loads(Path(args.samples).read_text(encoding='utf-8'))['samples']
    chosen = [s for s in samples if s['id'] in set(args.ids)] if args.ids else samples[:args.num]
    if not chosen:
        raise SystemExit('no samples selected')

    columns = [('Image', None, 'image'), ('Mask', None, 'mask')]
    for spec in args.methods:
        name, path = spec.split('=', 1)
        columns.append((name, Path(path), 'baseline'))
    if args.ours:
        name, path = args.ours.split('=', 1)
        columns.append((name, Path(path), 'ours'))

    ncol = len(columns)
    surf_dirs = {}
    for spec in args.surf_maps:
        k, v = spec.split('=', 1)
        surf_dirs[k] = Path(v)
    fig_w = args.fig_width
    cell_w = fig_w / ncol
    head_h = 0.19
    img_h = cell_w * (1.0 if args.roi_panels else 417.0 / 622.0)
    surf_h = cell_w * (0.84 if args.roi_panels else 0.94)
    gap_h = 0.10
    fig_h = head_h + len(chosen) * (img_h + surf_h + gap_h) + 0.06

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor='white')
    heights = [head_h]
    for _ in chosen:
        heights += [img_h, surf_h, gap_h]
    gs = gridspec.GridSpec(len(heights), ncol, figure=fig, height_ratios=heights,
                           hspace=0.015, wspace=0.004,
                           left=0.003, right=0.997, top=0.998, bottom=0.003)

    for j, (name, _, kind) in enumerate(columns):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(HEADER_COLORS[kind])
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('white'), spine.set_linewidth(0.8)
        ax.text(0.5, 0.5, name, ha='center', va='center', fontsize=args.header_font,
                fontweight='bold', color='black' if kind in ('image', 'mask') else 'white')

    for si, sample in enumerate(chosen):
        stem = sample['id']
        gt = cv2.imread(str(SEAL / 'Seal_data/test/GT' / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f'missing GT {stem}')
        img = load_image(stem, args.image, args.image_gamma, args.image_stretch)
        img_raw = load_image(stem, args.image, 1.0, 'none')
        roi = roi_from_gt(gt, margin=args.roi_margin, min_box=args.roi_min)
        y0, y1, x0, x1 = roi
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = args.crop3d // 2
        sy0, sy1 = max(0, cy - half), min(gt.shape[0], cy + half)
        sx0, sx1 = max(0, cx - half), min(gt.shape[1], cx + half)

        panels, surfaces = [], []
        for name, path, kind in columns:
            if kind == 'image':
                if args.roi_panels:
                    panels.append(compose_panel(img, roi, with_box=True, with_inset=False))
                else:
                    panels.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                surfaces.append(img[sy0:sy1, sx0:sx1])
            elif kind == 'mask':
                gt_bin = gt > 127
                sheet = np.where(gt_bin, 255, 0).astype(np.uint8)
                if args.diag:
                    panels.append(diag_panel(gt_bin, gt_bin, gt, roi, args.roi_panels))
                else:
                    panels.append(roi_panel(sheet, roi) if args.roi_panels else compose_panel(sheet, roi))
                surfaces.append(sheet[sy0:sy1, sx0:sx1])
            else:
                m = load_map(path / f'{stem}.png')
                if m.shape != gt.shape:
                    raise ValueError(f'{name}/{stem}: {m.shape} != {gt.shape}')
                pred_bin = m >= 128
                sheet = np.where(pred_bin, 255, 0).astype(np.uint8)
                if args.diag:
                    panels.append(diag_panel(pred_bin, gt > 127, gt, roi, args.roi_panels))
                else:
                    panels.append(roi_panel(sheet, roi) if args.roi_panels else compose_panel(sheet, roi))
                if name in surf_dirs:
                    resp = load_map(surf_dirs[name] / f'{stem}.png')
                    if resp.shape != gt.shape:
                        raise ValueError(f'{name}/{stem}: response {resp.shape} != {gt.shape}')
                else:
                    resp = m
                surfaces.append(resp[sy0:sy1, sx0:sx1])

        r_img = 1 + si * 3
        r_srf = r_img + 1
        for j, panel in enumerate(panels):
            ax = fig.add_subplot(gs[r_img, j])
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB), interpolation='nearest')
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('black'), spine.set_linewidth(0.3)
        for j, surf in enumerate(surfaces):
            cmap = 'gray' if columns[j][2] == 'image' else SURF_CMAP
            smooth = 5.0 if columns[j][2] == 'image' else 0.9
            arr = gaussian_filter(surf.astype(np.float32), smooth)
            peak = float(arr.max())
            vmax = 255.0 if args.z_mode == 'shared' else max(20.0, peak)
            col = safe_name(columns[j][0])
            p3d = cache_root / f'{stem}_{j:02d}_{col}_{int(round(vmax))}.png'
            if not p3d.exists():
                render_surface_png(arr, p3d, vmax=vmax, cmap=cmap)
            if split_root is not None:
                dst = split_root / f'{si + 1:02d}_{stem}' / '3d' / f'{j:02d}_{col}.png'
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p3d, dst)
            # 贴回格子并铺满（PNG 已裁掉白边，aspect='auto' 让 3D 盒子占满整格）
            ax = fig.add_subplot(gs[r_srf, j])
            ax.imshow(cv2.cvtColor(cv2.imread(str(p3d)), cv2.COLOR_BGR2RGB),
                      aspect='auto', interpolation='lanczos')
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        if split_root is not None:
            sdir = split_root / f'{si + 1:02d}_{stem}' / '2d'
            sdir.mkdir(parents=True, exist_ok=True)
            for j, panel in enumerate(panels):
                cv2.imwrite(str(sdir / f'{j:02d}_{safe_name(columns[j][0])}.png'), panel)

    png = out / f'{args.tag}_qualitative.png'
    pdf = out / f'{args.tag}_qualitative.pdf'
    fig.savefig(png, dpi=args.dpi, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    meta = {
        'figure': str(png), 'pdf': str(pdf),
        'columns': [c[0] for c in columns],
        'samples': [s['id'] for s in chosen],
        'image_modality': args.image, 'image_gamma': args.image_gamma,
        'image_stretch': args.image_stretch,
        'threshold': 'pred >= 128 (prob > 0.5)',
        'crop3d': args.crop3d, 'z_mode': args.z_mode,
        'surf_maps': {k: str(v) for k, v in surf_dirs.items()},
        'roi_margin_px': args.roi_margin, 'min_roi_px': args.roi_min, 'inset_zoom': ZOOM,
        'roi_panels': bool(args.roi_panels),
        'split_dir': str(split_root) if split_root is not None else None,
        'header_colors': HEADER_COLORS,
    }
    (out / f'{args.tag}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
