#!/usr/bin/env python3
"""Write the fully-supervised IRSTD baseline rows into docs/Seal量化SOTA表.xlsx.

Also replaces the NESS-Net row (17) with the public-weight cross-domain result that
AGENTS.md requires (the previous row held the Seal box-adaptation epoch-5 numbers).

Usage:
    python update_table.py --xlsx docs/Seal量化SOTA表.xlsx --metrics <dir with <Name>.json>
"""
import argparse
import json
from copy import copy
from pathlib import Path

from openpyxl import load_workbook

ROWS = [
    dict(name='DNA-Net', json='DNANet', source='TIP22', modal='T', supervision='Full GT',
         paper='Dense Nested Attention Network for Infrared Small Target Detection',
         code='https://github.com/YeRen123455/Infrared-Small-Target-Detection',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='MSHNet', json='MSHNet', source='CVPR24', modal='T', supervision='Full GT',
         paper='Infrared Small Target Detection with Scale and Location Sensitivity',
         code='https://github.com/Lliu666/MSHNet',
         note='Seal train 全监督重训 → Seal test'),
    dict(name='UIU-Net', json='UIU-Net', source='TIP23', modal='T', supervision='Full GT',
         paper='UIU-Net: U-Net in U-Net for Infrared Small Object Detection',
         code='https://github.com/danfenghong/IEEE_TIP_UIU-Net',
         note='Seal train 全监督重训 → Seal test'),
]

NESS = dict(S=0.4666185279426068, Fw=0.001733883532895621, E=0.263767112039326,
            MAE=0.042296525090932846, IoU=0.0007333271181027003, F1=0.001465579487023831,
            Pd=0.005146959230665041, Fa=0.032954249624511)

HEADER = {}


def find_columns(ws):
    """Locate metric and metadata columns; headers live in rows 1 and 2."""
    cols = {}
    for row in (1, 2):
        for col in range(1, ws.max_column + 1):
            value = ws.cell(row=row, column=col).value
            if value in ('S↑', 'Fw↑', 'E↑', 'MAE↓', 'IoU', 'F1', 'Pd', 'Fa'):
                cols[value.rstrip('↑↓')] = col
            if value in ('Method', 'Source', 'Modal', 'Supervision', '论文', '代码', '重训/测试', '备注'):
                HEADER.setdefault(value, col)
    missing = {'S', 'Fw', 'E', 'MAE', 'IoU', 'F1', 'Pd', 'Fa'} - set(cols)
    if missing:
        raise SystemExit(f'cannot locate metric columns: {sorted(missing)}')
    return cols


def copy_row_style(ws, src_row, dst_row, ncols):
    for col in range(1, ncols + 1):
        src = ws.cell(row=src_row, column=col)
        dst = ws.cell(row=dst_row, column=col)
        if src.has_style:
            dst._style = copy(src._style)


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
    print('metric columns', cols)
    print('header columns', HEADER)

    # --- NESS-Net row: public-weight cross-domain result (AGENTS.md requirement) ---
    for row in range(3, ws.max_row + 1):
        value = ws.cell(row=row, column=HEADER['Method']).value or ''
        if str(value).startswith('NESS-Net'):
            for key, metric in NESS.items():
                ws.cell(row=row, column=cols[key]).value = metric
            ws.cell(row=row, column=HEADER['Method']).value = 'NESS-Net（公开权重跨域测试）'
            ws.cell(row=row, column=HEADER['重训/测试']).value = '公开 refinement + NAMLab 跨域测试（原函数）'
            print(f'row {row}: NESS-Net replaced with cross-domain metrics')
            break

    # --- fully-supervised baselines appended after row 20 ---
    start = 21
    for offset, spec in enumerate(ROWS):
        row = start + offset
        record = json.loads((metrics_dir / f"{spec['json']}.json").read_text(encoding='utf-8'))
        m = record['eight_metrics']
        copy_row_style(ws, 20, row, ws.max_column)
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
        print('dry run: nothing written')
        return
    wb.save(args.xlsx)
    print('saved', args.xlsx)


if __name__ == '__main__':
    main()
