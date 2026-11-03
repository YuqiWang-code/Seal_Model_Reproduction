"""Verify the eight full-supervision rows against their full-precision artifacts.

Checks, per method:
  1. eight_metrics.json values equal the xlsx row (bit-level tolerance 1e-12),
  2. IoU/F1/Pd/Fa recomputed from small_metric_counts,
  3. counts are the same GT/test batch as the rest of the project
     (n_images 1983, n_gt 7383, total_pixels 514338642),
  4. fullsup_metrics.json (four original metrics) agrees with eight_metrics.json.
"""

import json
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['S', 'Fw', 'E', 'MAE', 'IoU', 'F1', 'Pd', 'Fa']
HEADERS = {'S↑': 'S', 'Fw↑': 'Fw', 'E↑': 'E', 'MAE↓': 'MAE',
           'IoU': 'IoU', 'F1': 'F1', 'Pd': 'Pd', 'Fa': 'Fa'}
ART = {
    'DNA-Net': 'DNANet_Seal评估产物',
    'MSHNet': 'MSHNet_Seal评估产物',
    'UIU-Net': 'UIU-Net_Seal评估产物',
    'SCTransNet': 'SCTransNet_Seal评估产物',
    'RPCANet': 'RPCANet_Seal评估产物',
    'HDNet': 'HDNet_Seal评估产物',
    'ILNet': 'ILNet_Seal评估产物',
    'RPCANet++': 'RPCANetpp_Seal评估产物',
}


def table():
    ws = openpyxl.load_workbook(ROOT / 'docs' / 'Seal量化SOTA表.xlsx', data_only=True).worksheets[0]
    col = {}
    for r in (1, 2):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and v.strip() in HEADERS:
                col[HEADERS[v.strip()]] = c
    return {ws.cell(r, 1).value.strip(): {k: ws.cell(r, c).value for k, c in col.items()}
            for r in range(21, 29)}


def main():
    rows = table()
    problems = []
    for name, row in rows.items():
        data = json.loads((ROOT / 'docs' / ART[name] / 'eight_metrics.json').read_text(encoding='utf-8'))
        em = data['eight_metrics']
        counts = data['small_metric_counts']
        for key in KEYS:
            if abs(em[key] - row[key]) > 1e-12:
                problems.append(f'{name}.{key}: table={row[key]!r} artifact={em[key]!r}')
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        union = tp + fp + fn
        f1d = 2 * tp + fp + fn
        recalc = {
            'IoU': tp / union,
            'F1': 2 * tp / f1d,
            'Pd': counts['n_detected'] / counts['n_gt'],
            'Fa': counts['false_alarm_pixels'] / counts['total_pixels'],
        }
        for key, value in recalc.items():
            if abs(value - em[key]) > 1e-12:
                problems.append(f'{name}.{key}: stored={em[key]!r} recalc={value!r}')
        if (counts['n_images'], counts['n_gt'], counts['total_pixels']) != (1983, 7383, 514338642):
            problems.append(f'{name}: unexpected batch {counts["n_images"]}/{counts["n_gt"]}/{counts["total_pixels"]}')
        if counts['n_detected'] > counts['n_pred']:
            problems.append(f'{name}: n_detected > n_pred')
        four = json.loads((ROOT / 'docs' / ART[name] / 'fullsup_metrics.json').read_text(encoding='utf-8'))
        flat = four.get('Seal', four)
        for src, dst in (('S', 'S'), ('F', 'Fw'), ('E', 'E'), ('MAE', 'MAE')):
            if src in flat and abs(float(flat[src]) - em[dst]) > 1e-9:
                problems.append(f'{name}.{dst}: fullsup_metrics={flat[src]!r} artifact={em[dst]!r}')
        method = data.get('method', '?')
        print(f"{name:11s} row={row['IoU']:.8f} artifact_method={method:10s} protocol={data['protocol'][:40]}")
    print(f'\n{len(problems)} problem(s)')
    for p in problems:
        print(' -', p)


if __name__ == '__main__':
    main()
