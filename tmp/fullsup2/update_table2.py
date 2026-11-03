#!/usr/bin/env python3
"""Append the 2024/2025/2026 fully-supervised baselines to docs/Seal量化SOTA表.xlsx.

Rows 21-23 (DNA-Net / MSHNet / UIU-Net) are left untouched; the new methods go
into rows 24+ in the same layout, with styles copied from the last filled row.
"""
import argparse
import json
from copy import copy
from pathlib import Path

from openpyxl import load_workbook

ROWS = [
    dict(name='SCTransNet', json='SCTransNet', source='TGRS24', modal='T', supervision='Full GT',
         paper='SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection',
         code='https://github.com/xdFai/SCTransNet',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='RPCANet', json='RPCANet', source='WACV24', modal='T', supervision='Full GT',
         paper='RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection',
         code='https://github.com/fengyiwu98/RPCANet',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='HDNet', json='HDNet', source='TGRS25', modal='T', supervision='Full GT',
         paper='HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection',
         code='https://github.com/xumingzhu989/HDNet-TGRS',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='ILNet', json='ILNet', source='TAES25', modal='T', supervision='Full GT',
         paper='Low-Level Matters for Salient Infrared Small Target Detection',
         code='https://github.com/Li-Haoqing/ILNet',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='RPCANet++', json='RPCANetpp', source='TPAMI26', modal='T', supervision='Full GT',
         paper='RPCANet++: Deep Interpretable Robust PCA for Sparse Object Segmentation',
         code='https://github.com/fengyiwu98/RPCANet',
         note='Seal train 全监督重训 → Seal test'),
]

HEADER = {}


def find_columns(ws):
    cols = {}
    for row in (1, 2):
        for col in range(1, ws.max_column + 1):
            value = ws.cell(row=row, column=col).value
            if value in ('S↑', 'Fw↑', 'E↑', 'MAE↓', 'IoU', 'F1', 'Pd', 'Fa'):
                cols[value.rstrip('↑↓')] = col
            if value in ('Method', 'Source', 'Modal', 'Supervision', '论文', '代码', '重训/测试'):
                HEADER.setdefault(value, col)
    missing = {'S', 'Fw', 'E', 'MAE', 'IoU', 'F1', 'Pd', 'Fa'} - set(cols)
    if missing:
        raise SystemExit(f'cannot locate metric columns: {sorted(missing)}')
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default='docs/Seal量化SOTA表.xlsx')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    metrics_dir = Path(args.metrics)
    wb = load_workbook(args.xlsx)
    ws = wb.active
    cols = find_columns(ws)
    last = ws.max_row
    while last > 1 and ws.cell(row=last, column=1).value in (None, ''):
        last -= 1
    print('current last row', last, 'columns', cols)

    start = last + 1
    for offset, spec in enumerate(ROWS):
        row = start + offset
        record = json.loads((metrics_dir / f"{spec['json']}.json").read_text(encoding='utf-8'))
        m = record['eight_metrics']
        for col in range(1, ws.max_column + 1):
            src = ws.cell(row=last, column=col)
            if src.has_style:
                ws.cell(row=row, column=col)._style = copy(src._style)
        ws.cell(row=row, column=HEADER['Method']).value = spec['name']
        ws.cell(row=row, column=HEADER['Source']).value = spec['source']
        ws.cell(row=row, column=HEADER['Modal']).value = spec['modal']
        ws.cell(row=row, column=HEADER['Supervision']).value = spec['supervision']
        for key in ('S', 'Fw', 'E', 'MAE', 'IoU', 'F1', 'Pd', 'Fa'):
            ws.cell(row=row, column=cols[key]).value = m[key]
        ws.cell(row=row, column=HEADER['论文']).value = spec['paper']
        ws.cell(row=row, column=HEADER['代码']).value = spec['code']
        ws.cell(row=row, column=HEADER['重训/测试']).value = spec['note']
        print(f'row {row}: {spec["name"]} {m}')

    if args.dry_run:
        print('dry run, nothing written')
        return
    wb.save(args.xlsx)
    print('saved', args.xlsx)


if __name__ == '__main__':
    main()
