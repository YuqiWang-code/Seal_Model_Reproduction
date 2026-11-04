#!/usr/bin/env python3
"""Rank Seal test images for qualitative figure selection and build a review page.

Per-image scores are computed from the 0/255 prediction maps (threshold >= 128,
the project rule). The ranking metric is the *per-image advantage over the
baselines*:

    score = F1(Ours) - mean_k F1(baseline_k)

so the top of the list shows frames where Ours clearly beats every baseline while
the baselines disagree (missed targets / false alarms). Ties break on F1(Ours),
then IoU(Ours). Nothing about the test set is used to choose weights; this only
orders samples for a human to pick from.

Outputs a JSON with all per-image numbers and a self-contained HTML page with
base64 thumbnails (Microsoft YaHei font), where rows can be clicked to collect IDs.

Usage
-----
    python tools/rank_qualitative_samples.py --top 200 \
        --methods DNA-Net=results/DNANet/Seal/test_binary \
                  MSHNet=results/MSHNet/Seal/test_binary \
                  RPCANet=results/RPCANet/Seal/test_binary \
                  HDNet=results/HDNet/Seal/test_binary \
                  L2SKNet=results/L2SKNet/Seal/test_binary \
                  BGM=results/BGM/Seal/test_binary \
                  RPCANet++=results/RPCANetpp/Seal/test_binary \
        --ours Ours=Ours/test_maps \
        --out-json docs/qualitative_rank_top200.json \
        --out-html docs/qualitative_review_top200.html
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

SEAL = Path('/share_datasets/Seal_Model_Reproduction')
CROP = 128          # thumbnail crop side (before zoom)
THUMB = 96          # saved thumbnail side
JPEG_Q = 70


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'cannot read {path}')
    return img


def pixel_scores(pred_bin, gt_bin):
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return f1, iou, tp, fp, fn


def crop_thumb(img, cy, cx, side=CROP, out=THUMB, color=False):
    h, w = img.shape[:2]
    half = side // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (out, out), interpolation=cv2.INTER_NEAREST)
    if not color:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', required=True, help='Name=dir (binary or prob maps)')
    ap.add_argument('--ours', default='Ours=Ours/test_maps')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--gt-dir', default=str(SEAL / 'Seal_data/test/GT'))
    ap.add_argument('--out-json', default=str(SEAL / 'docs/qualitative_rank_top200.json'))
    ap.add_argument('--out-html', default=str(SEAL / 'docs/qualitative_review_top200.html'))
    ap.add_argument('--no-html', action='store_true')
    args = ap.parse_args()

    methods = [m.split('=', 1) for m in args.methods]
    ours_name, ours_dir = args.ours.split('=', 1)
    gt_dir = Path(args.gt_dir)
    stems = sorted(p.stem for p in gt_dir.glob('*.png'))
    print(f'{len(stems)} test frames, {len(methods)} baselines + {ours_name}')

    records = []
    for stem in stems:
        gt = load_gray(gt_dir / f'{stem}.png')
        gt_bin = gt > 127
        ys, xs = np.where(gt_bin)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        rec = {'id': stem, 'gt_px': int(gt_bin.sum()),
               'gt_components': int(cv2.connectedComponents(gt_bin.astype(np.uint8), 8)[0] - 1),
               'method_scores': {}}

        ours_path = Path(ours_dir) / f'{stem}.png'
        ours_map = load_gray(ours_path)
        ours_bin = ours_map >= 128
        f1, iou, tp, fp, fn = pixel_scores(ours_bin, gt_bin)
        rec['ours'] = {'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn,
                       'map': ours_map, 'bin': ours_bin}

        for name, d in methods:
            m = load_gray(Path(d) / f'{stem}.png')
            if m.shape != gt.shape:
                raise ValueError(f'{name}/{stem}: {m.shape} != {gt.shape}')
            mb = m >= 128
            f1b, ioub, tpb, fpb, fnb = pixel_scores(mb, gt_bin)
            rec['method_scores'][name] = {'f1': f1b, 'iou': ioub, 'tp': tpb, 'fp': fpb, 'fn': fnb,
                                          'map': m, 'bin': mb}

        base_f1 = [v['f1'] for v in rec['method_scores'].values()]
        base_iou = [v['iou'] for v in rec['method_scores'].values()]
        rec['base_f1_mean'] = float(np.mean(base_f1)) if base_f1 else 0.0
        rec['base_f1_max'] = float(np.max(base_f1)) if base_f1 else 0.0
        rec['base_iou_mean'] = float(np.mean(base_iou)) if base_iou else 0.0
        rec['score'] = float(rec['ours']['f1'] - rec['base_f1_mean'])
        rec['margin'] = float(rec['ours']['f1'] - rec['base_f1_max'])
        records.append(rec)

    records.sort(key=lambda r: (r['score'], r['ours']['f1'], r['ours']['iou']), reverse=True)
    top = records[:args.top]

    # thumbnails (centered on the GT centroid, 128 px window)
    for rank, rec in enumerate(top, 1):
        stem = rec['id']
        gt = load_gray(gt_dir / f'{stem}.png')
        ys, xs = np.where(gt > 127)
        cy, cx = int(ys.mean()), int(xs.mean())
        raw = cv2.imread(str(SEAL / 'Seal_data/test/T' / f'{stem}.tif'), cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi - lo > 1:
            img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        rec['rank'] = rank
        ours_thumb = crop_thumb(np.where(rec['ours']['bin'], 255, 0).astype(np.uint8), cy, cx)
        rec['thumb'] = {
            'image': crop_thumb(img, cy, cx),
            'mask': crop_thumb(np.where(gt > 127, 255, 0).astype(np.uint8), cy, cx),
            'ours': ours_thumb,
            ours_name: ours_thumb,
        }
        for name, _ in methods:
            rec['thumb'][name] = crop_thumb(
                np.where(rec['method_scores'][name]['bin'], 255, 0).astype(np.uint8), cy, cx)

    def clean_scores(d):
        return {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in d.items() if k not in ('map', 'bin')}

    out = {
        'protocol': {
            'threshold': 'pred >= 128 (prob > 0.5)',
            'metric': 'score = F1(Ours) - mean_k F1(baseline_k), per image',
            'tiebreak': 'F1(Ours) then IoU(Ours)',
            'scanned': len(records), 'kept': len(top),
            'baselines': [n for n, _ in methods], 'ours': ours_name,
            'note': 'ordering is for human sample selection only; no model/weight selection involved',
        },
        'samples': [{
            'rank': r['rank'], 'id': r['id'], 'score': round(r['score'], 6), 'margin': round(r['margin'], 6),
            'gt_px': r['gt_px'], 'gt_components': r['gt_components'],
            'ours': clean_scores(r['ours']),
            'base_f1_mean': round(r['base_f1_mean'], 6), 'base_f1_max': round(r['base_f1_max'], 6),
            'base_iou_mean': round(r['base_iou_mean'], 6),
            'methods': {n: clean_scores(r['method_scores'][n]) for n, _ in methods},
        } for r in top],
        'per_method': {n: [{'id': r['id'], **clean_scores(r['method_scores'][n])} for r in top]
                       for n, _ in methods},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('json', args.out_json)

    if args.no_html:
        return

    names = [n for n, _ in methods]
    cols = [('image', 'Image'), ('mask', 'Mask'), (ours_name, ours_name)] + [(n, n) for n in names]
    head = ''.join(f'<th>{label}</th>' for _, label in cols)
    rows = []
    for rec in top:
        cells = ''.join(f'<td><img src="{rec["thumb"][key]}" width="88" height="88"></td>'
                        for key, _ in cols)
        score_cells = ''.join(
            f'<td class="num" data-v="{rec["method_scores"][n]["f1"]:.4f}">{rec["method_scores"][n]["f1"]:.3f}'
            f'<span class="iou">/{rec["method_scores"][n]["iou"]:.3f}</span></td>'
            for n in names
        )
        rows.append(
            f'<tr data-id="{rec["id"]}" data-score="{rec["score"]:.4f}">'
            f'<td class="rank">{rec["rank"]}</td>'
            f'<td class="id">{rec["id"]}</td>'
            f'<td class="num">{rec["score"]:+.3f}</td>'
            f'<td class="num">{rec["margin"]:+.3f}</td>'
            f'<td class="num">{rec["ours"]["f1"]:.3f}<span class="iou">/{rec["ours"]["iou"]:.3f}</span></td>'
            f'{score_cells}<td class="num">{rec["gt_px"]}</td><td class="num">{rec["gt_components"]}</td>'
            f'{cells}</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Seal 定性样例 Top {len(top)}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "微软雅黑", sans-serif; margin: 16px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  p.meta {{ font-size: 12px; color: #555; margin: 4px 0 12px; }}
  table {{ border-collapse: collapse; background: #fff; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 2px 4px; text-align: center; white-space: nowrap; }}
  th {{ background: #eef3f8; position: sticky; top: 0; cursor: pointer; font-size: 11px; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td .iou {{ color: #888; font-size: 10px; }}
  td.id {{ font-weight: bold; }}
  tr:hover {{ background: #fff8e1; }}
  tr.sel {{ background: #d9ecff; outline: 2px solid #2e75b6; }}
  #bar {{ position: sticky; bottom: 0; background: #fff; border-top: 2px solid #2e75b6; padding: 8px; }}
  #ids {{ width: 100%; height: 44px; font-family: Consolas, monospace; font-size: 12px; }}
  button {{ font-family: inherit; font-size: 12px; padding: 4px 10px; margin-right: 8px; cursor: pointer; }}
</style></head><body>
<h1>Seal 定性样例候选 Top {len(top)}</h1>
<p class="meta">排序指标：<b>score = F1(Ours) − 基线平均 F1</b>（逐图）；并列时按 F1(Ours)、IoU(Ours)。
每行数字为 <b>F1</b><span style="color:#888"> / IoU</span>，阈值 prob &gt; 0.5。共扫描 {len(records)} 张，取前 {len(top)}。
点击行可选中/取消，底部会汇总 ID，复制后发我即可。</p>
<table id="t">
<thead><tr><th>#</th><th>ID</th><th>score</th><th>margin</th><th>{ours_name}</th>{''.join(f'<th>{n}</th>' for n in names)}<th>GT px</th><th>目标数</th>{head}</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<div id="bar">
  <button onclick="copyIds()">复制选中 ID</button>
  <button onclick="clearSel()">清空</button>
  <span id="cnt">已选 0</span>
  <textarea id="ids" placeholder="选中的 ID 会出现在这里"></textarea>
</div>
<script>
const rows = [...document.querySelectorAll('#t tbody tr')];
const sel = new Set();
function refresh() {{
  document.getElementById('ids').value = [...sel].join(' ');
  document.getElementById('cnt').textContent = '已选 ' + sel.size;
}}
rows.forEach(tr => tr.addEventListener('click', () => {{
  const id = tr.dataset.id;
  if (sel.has(id)) {{ sel.delete(id); tr.classList.remove('sel'); }}
  else {{ sel.add(id); tr.classList.add('sel'); }}
  refresh();
}}));
function clearSel() {{ sel.clear(); rows.forEach(r => r.classList.remove('sel')); refresh(); }}
function copyIds() {{
  const t = document.getElementById('ids');
  t.select(); document.execCommand('copy');
  alert('已复制 ' + sel.size + ' 个 ID');
}}
// click a header cell to sort by that column (numeric where possible)
document.querySelectorAll('#t thead th').forEach((th, idx) => th.addEventListener('click', () => {{
  const body = document.querySelector('#t tbody');
  const sorted = [...body.querySelectorAll('tr')].sort((a, b) => {{
    const av = a.children[idx].innerText.trim(), bv = b.children[idx].innerText.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    return (isNaN(an) || isNaN(bn)) ? av.localeCompare(bv) : bn - an;
  }});
  sorted.forEach(r => body.appendChild(r));
}}));
</script></body></html>"""
    Path(args.out_html).write_text(html, encoding='utf-8')
    print('html', args.out_html, Path(args.out_html).stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
