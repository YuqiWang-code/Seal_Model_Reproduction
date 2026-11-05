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
        --out …/Ours/qualitative --tag seal
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

# 3D 响应面配色：浅蓝底（亮、不压抑）+ 暖色峰（金 → 橙 → 红）
SURF_CMAP = LinearSegmentedColormap.from_list('seal3d', [
    (0.00, '#d3e5f7'),
    (0.32, '#a6cbe9'),
    (0.52, '#5fb0d8'),
    (0.66, '#f2b04a'),
    (0.80, '#e8622e'),
    (1.00, '#c81e1e'),
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
    """Square ROI around the GT, always fully inside the image.

    The window is a `side x side` square with side = max(min_box, gt_span + 2*margin);
    when the target sits near a border the window slides inward instead of being clipped,
    so every panel is a complete square crop (no black padding bars).
    """
    ys, xs = np.where(gt > 127)
    h, w = gt.shape
    if len(ys) == 0:
        side = int(min(min_box, h, w))
        return 0, side, 0, side
    cy, cx = int(ys.mean()), int(xs.mean())
    span = max(int(ys.max() - ys.min()), int(xs.max() - xs.min()))
    side = int(min(max(min_box, span + 2 * margin), h, w))
    y0 = min(max(0, cy - side // 2), h - side)
    x0 = min(max(0, cx - side // 2), w - side)
    return y0, y0 + side, x0, x0 + side


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


def image_panel_roi(img, roi, target=384, mode='full', min_box=MIN_BOX, gt_box=None,
                    pad=4, draw_box=False):
    """Image column = the whole scene (the ROI is magnified in the next columns).

    draw_box=False (默认，v12 起) 不画任何框：Image 列就是原图，定位交给各列一致的视野。
    需要框时（`--image-box`）边长取 ``max(min_box, max(GT 宽, GT 高) + 2*pad)`` 换算到面板坐标，
    以 GT 包围盒为中心并保证 GT 矩形整体落在框内。
    """
    h, w = img.shape[:2]
    y0, y1, x0, x1 = roi

    if mode == 'crop417':
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = min(h, w)
        x0c = int(min(max(0, round(cx - side / 2.0)), w - side))
        y0c = int(min(max(0, round(cy - side / 2.0)), h - side))
        crop = img[y0c:y0c + side, x0c:x0c + side]
        panel = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), (target, target),
                           interpolation=cv2.INTER_AREA)
        k = target / float(side)
        bx0, bx1 = int(round((x0 - x0c) * k)), int(round((x1 - x0c) * k))
        by0, by1 = int(round((y0 - y0c) * k)), int(round((y1 - y0c) * k))
        if draw_box:
            cv2.rectangle(panel, (bx0, by0), (bx1, by1), RED, 3, cv2.LINE_AA)
        return panel

    panel = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (target, target),
                       interpolation=cv2.INTER_AREA)
    if not draw_box:
        return panel
    sx, sy = target / float(w), target / float(h)
    uni = min_box * (sx * sy) ** 0.5          # 统一框边长下限（64 px ROI → 约 48 px）

    if gt_box is None:
        gx0, gx1 = x0 * sx, x1 * sx
        gy0, gy1 = y0 * sy, y1 * sy
    else:
        gx0, gx1 = gt_box[0] * sx, (gt_box[2] + 1) * sx
        gy0, gy1 = gt_box[1] * sy, (gt_box[3] + 1) * sy
    gx1 = min(gx1, target - 1.0)
    gy1 = min(gy1, target - 1.0)

    side = max(uni, (gx1 - gx0) + 2 * pad, (gy1 - gy0) + 2 * pad)
    side = min(side, float(target - 1))
    half = side / 2.0

    def place(lo, hi):
        b0 = int(round((lo + hi) / 2.0 - half))
        b0 = min(max(b0, 0), int(round(target - 1 - side)))
        return b0, b0 + int(round(side))

    bx0, bx1 = place(gx0, gx1)
    by0, by1 = place(gy0, gy1)
    # 兜底：保证 GT 矩形完全落在框内
    bx0, bx1 = min(bx0, int(np.floor(gx0))), max(bx1, int(np.ceil(gx1)))
    by0, by1 = min(by0, int(np.floor(gy0))), max(by1, int(np.ceil(gy1)))
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(target - 1, bx1), min(target - 1, by1)
    cv2.rectangle(panel, (bx0, by0), (bx1, by1), RED, 3, cv2.LINE_AA)
    return panel


def roi_crop_canvas(img_bgr, roi, target=384):
    """Crop to the ROI, upscale nearest-neighbour, pad to a square canvas."""
    y0, y1, x0, x1 = roi
    crop = img_bgr[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    k = max(1, int(round(target / max(1, side))))
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    ch, cw = crop.shape[:2]
    canvas = np.zeros((max(ch, cw), max(ch, cw), 3), np.uint8)
    oy, ox = (canvas.shape[0] - ch) // 2, (canvas.shape[1] - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas, k, ox, oy


def draw_gt_boxes(canvas, gt_bin, roi, k, ox, oy, box_pad=8):
    """One red locator box per GT connected component.

    Single target -> 36% of the panel side; several targets -> 0.45/n so they do not
    merge. The box only says "the target is here" — it is not a second magnification.
    """
    y0, y1, x0, x1 = roi
    h, w = gt_bin.shape
    n_lab, labels = cv2.connectedComponents(gt_bin.astype(np.uint8), connectivity=8)
    comps = []
    for lab in range(1, n_lab):
        ys2, xs2 = np.where(labels == lab)
        if len(ys2):
            comps.append((int(xs2.min()), int(xs2.max()), int(ys2.min()), int(ys2.max())))
    if not comps:
        comps = [(x0, x1, y0, y1)]
    frac = 0.36 if len(comps) == 1 else max(0.12, min(0.26, 0.45 / len(comps)))
    H, W = canvas.shape[:2]
    for (cx0, cx1, cy0, cy1) in comps:
        cx = ox + ((cx0 + cx1) / 2.0 - x0) * k
        cy = oy + ((cy0 + cy1) / 2.0 - y0) * k
        span = max(frac * W, (cx1 - cx0 + 1) * k, (cy1 - cy0 + 1) * k)
        half = span / 2.0 + box_pad
        bx0, bx1 = max(0, int(round(cx - half))), min(W - 1, int(round(cx + half)))
        by0, by1 = max(0, int(round(cy - half))), min(H - 1, int(round(cy + half)))
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), RED, 3, cv2.LINE_AA)
    return canvas


def diag_panel(pred_bin, gt_bin, roi, roi_mode, target=384, box_pad=8, draw_box=False):
    """TP/FP/FN colour panel: TP white, FP red, FN green.

    Used for the Visio assembly so each method's error pattern (miss = green,
    false alarm = red) is visible at a glance. `draw_box=False` keeps the panels clean —
    the only red box in the figure is the one in the Image column marking the crop.
    """
    h, w = gt_bin.shape
    colour = np.zeros((h, w, 3), np.uint8)
    colour[pred_bin & gt_bin] = (255, 255, 255)
    colour[pred_bin & ~gt_bin] = (0, 0, 255)      # BGR red
    colour[~pred_bin & gt_bin] = (0, 255, 0)      # BGR green

    y0, y1, x0, x1 = roi
    if roi_mode:
        canvas, k, ox, oy = roi_crop_canvas(colour, roi, target)
        if draw_box:
            return draw_gt_boxes(canvas, gt_bin, roi, k, ox, oy, box_pad)
        return canvas

    # 整幅模式：每个目标一个框（draw_box=False 时只返回彩色图）
    panel = colour.copy()
    if not draw_box:
        return panel
    n_lab, labels = cv2.connectedComponents(gt_bin.astype(np.uint8), connectivity=8)
    boxes = []
    for lab in range(1, n_lab):
        ys2, xs2 = np.where(labels == lab)
        if len(ys2) == 0:
            continue
        boxes.append((int(xs2.min()), int(xs2.max()), int(ys2.min()), int(ys2.max())))
    if not boxes:
        boxes = [(x0, x1, y0, y1)]
    for (cx0, cx1, cy0, cy1) in boxes:
        cv2.rectangle(panel, (max(0, cx0 - box_pad), max(0, cy0 - box_pad)),
                      (min(w - 1, cx1 + box_pad), min(h - 1, cy1 + box_pad)),
                      RED, 2, cv2.LINE_AA)
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


def render_surface_png(arr, path, vmax, cmap='jet', gamma=0.45,
                       size_in=(0.873, 0.733), dpi=1200, pad=14):
    """One 3D box as a standalone PNG, tight-cropped so the box fills the frame.

    `arr` is the already-smoothed response crop. Rendered at one cell of the composite
    figure (0.873 x 0.733 in) with the same tick font, then the white border is trimmed
    to `pad` px, so pasting it back with aspect='auto' fills the cell.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = plt.figure(figsize=size_in, dpi=dpi, facecolor='white')
    ax = f.add_subplot(111, projection='3d')
    add_surface(ax, arr, vmax=vmax, cmap=cmap, gamma=gamma)
    f.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    import io

    buf = io.BytesIO()
    f.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(f)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = trim_white(img, pad=pad)
    cv2.imwrite(str(path), img)
    return path


def add_surface(ax, arr, vmax=255.0, cmap=SURF_CMAP, tick_font=2.2, gamma=0.40):
    """PDDiff Fig. 9 style surface with a legible wireframe and tick labels.

    `arr` must already be smoothed; `vmax` is normally the peak of `arr`, so the target
    response always reaches the top of the box. `gamma=None` switches the colour mapping
    to linear (used for the Image column, where the surface is an intensity map).
    """
    h, w = arr.shape
    step = max(1, min(h, w) // 48)
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    zz = arr[::step, ::step]
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax) if gamma else None
    if norm is None:
        ax.plot_surface(xx, yy, zz, cmap=cmap, vmin=0, vmax=vmax, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
    else:
        ax.plot_surface(xx, yy, zz, cmap=cmap, norm=norm, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
    ax.set_zlim(0, vmax)
    ax.view_init(elev=26, azim=-64)

    # 坐标刻度：X/Y 是裁剪块像素坐标，Z 是响应值；取整数"好刻度"，与 PDDiff 一致
    loc = MaxNLocator(nbins=3, integer=True, steps=[1, 2, 2.5, 5, 10])
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, steps=[1, 2, 2.5, 5, 10]))
    for axis in ('x', 'y'):
        ax.tick_params(axis=axis, labelsize=tick_font, pad=-3.0, length=0.9, width=0.4,
                       colors='0.10')
    ax.tick_params(axis='z', labelsize=tick_font, pad=-1.5, length=0.9, width=0.4,
                   colors='0.10')
    # 面板只有 0.87 in 宽，再放 X/Y/Z 轴名会和刻度数字撞在一起，只保留三向刻度数字
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 1))   # 外框要看得见
        axis.pane.set_linewidth(0.5)
        axis._axinfo['grid'].update(color=(0.62, 0.62, 0.62, 1), linewidth=0.45)
        # 去掉 matplotlib 默认的黑色粗轴线，改成细而清楚的灰线
        axis.line.set_color((0.30, 0.30, 0.30, 1))
        axis.line.set_linewidth(0.4)

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
    ap.add_argument('--image-mode', default='full', choices=['full', 'crop417'],
                    help='roi 版式的 Image 列：full = 整幅压成正方形；crop417 = 417x417 原比例窗口')
    ap.add_argument('--image-box', action='store_true',
                    help='在 Image 列画红框（v12 起默认不画；仅为兼容旧版式保留）')
    ap.add_argument('--diag', action='store_true',
                    help='2D panels = TP white / FP red / FN green + red GT box (for Visio)')
    ap.add_argument('--roi-panels', action='store_true',
                    help='2D row = upscaled ROI instead of full frame + inset')
    ap.add_argument('--z-mode', default='per-panel', choices=['shared', 'per-panel'])
    ap.add_argument('--out', default=str(SEAL / 'Ours' / 'qualitative'))
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
        _ys, _xs = np.where(gt > 127)
        gt_box = None if len(_ys) == 0 else (int(_xs.min()), int(_ys.min()),
                                             int(_xs.max()), int(_ys.max()))
        y0, y1, x0, x1 = roi
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = args.crop3d // 2
        sy0, sy1 = max(0, cy - half), min(gt.shape[0], cy + half)
        sx0, sx1 = max(0, cx - half), min(gt.shape[1], cx + half)

        panels, surfaces = [], []
        for name, path, kind in columns:
            if kind == 'image':
                if args.roi_panels:
                    # roi 版式：Image 列 = 整幅场景压缩成正方形 + 唯一一个正方形红框，
                    # 表示其余各列就是这块区域放大出来的（Mask 之后不再画框）
                    panels.append(image_panel_roi(img, roi, mode=args.image_mode,
                                                   min_box=args.roi_min, gt_box=gt_box,
                                                   draw_box=args.image_box))
                else:
                    panels.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                surfaces.append(img[sy0:sy1, sx0:sx1])
            elif kind == 'mask':
                gt_bin = gt > 127
                sheet = np.where(gt_bin, 255, 0).astype(np.uint8)
                if args.diag:
                    panels.append(diag_panel(gt_bin, gt_bin, roi, args.roi_panels))
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
                    panels.append(diag_panel(pred_bin, gt > 127, roi, args.roi_panels))
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
            is_img = columns[j][2] == 'image'
            cmap = SURF_CMAP if is_img else SURF_CMAP
            smooth = 5.0 if is_img else 0.9
            arr = gaussian_filter(surf.astype(np.float32), smooth)
            peak = float(arr.max())
            vmax = 255.0 if args.z_mode == 'shared' else max(20.0, peak)
            # Image 列：彩色（PDDiff 那样），用线性映射；预测列用 PowerNorm 提亮峰
            gamma = None if is_img else 0.40
            col = safe_name(columns[j][0])
            p3d = cache_root / f'{stem}_{j:02d}_{col}_{int(round(vmax))}_{"lin" if is_img else "pow"}.png'
            if not p3d.exists():
                render_surface_png(arr, p3d, vmax=vmax, cmap=cmap, gamma=gamma)
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
