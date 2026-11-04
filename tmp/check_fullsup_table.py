"""Cross-check the full-supervision rows (21-28) of docs/Seal量化SOTA表.xlsx.

Compared against AGENTS.md (normative, 5-decimal rendering) and against each
per-method experiment record (full precision). Columns are located by header
name, never by index.
"""

import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r'-?\d+\.\d+(?:[eE][-+]?\d+)?')
KEYS = ['S', 'Fw', 'E', 'MAE', 'IoU', 'F1', 'Pd', 'Fa']
EXTRA = ['Params', 'GFLOPs', 'FPS']
HEADERS = {'S↑': 'S', 'Fw↑': 'Fw', 'E↑': 'E', 'MAE↓': 'MAE',
           'IoU': 'IoU', 'F1': 'F1', 'Pd': 'Pd', 'Fa': 'Fa',
           'IoU↑': 'IoU', 'F1↑': 'F1', 'Pd↑': 'Pd',
           'Params↓': 'Params', 'GFLOPs↓': 'GFLOPs', 'FPS↑': 'FPS'}
DOCS = {
    'DNA-Net': 'DNANet_Seal实验记录.md',
    'MSHNet': 'MSHNet_Seal实验记录.md',
    'UIU-Net': 'UIU-Net_Seal实验记录.md',
    'SCTransNet': 'SCTransNet_Seal实验记录.md',
    'RPCANet': 'RPCANet_Seal实验记录.md',
    'HDNet': 'HDNet_Seal实验记录.md',
    'ILNet': 'ILNet_Seal实验记录.md',
    'RPCANet++': 'RPCANet++_Seal实验记录.md',
    'L2SKNet': 'L2SKNet_Seal实验记录.md',
    'BGM': 'BGM_Seal实验记录.md',
    'DRPCA-Net': 'DRPCA-Net_Seal实验记录.md',
    'DURD': 'DURD_Seal实验记录.md',
}


def read_table():
    ws = openpyxl.load_workbook(ROOT / 'docs' / 'Seal量化SOTA表.xlsx', data_only=True).worksheets[0]
    col = {}
    for r in (1, 2):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and v.strip() in HEADERS:
                col[HEADERS[v.strip()]] = c
    rows = {}
    for r in range(21, 33):
        name = ws.cell(r, 1).value.strip()
        rows[name] = {'row': r, **{k: ws.cell(r, c).value for k, c in col.items()}}
    return rows


def agents_rows():
    out = {}
    for line in (ROOT / 'AGENTS.md').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.strip('|').split('|')]
        if len(parts) not in (11, 14):
            continue
        try:
            values = [float(x) for x in parts[3:11]]
            epoch = int(parts[2])
        except ValueError:
            continue
        extra = {}
        if len(parts) == 14:
            try:
                extra = dict(zip(EXTRA, (float(x) for x in parts[11:14])))
            except ValueError:
                extra = {}
        out[parts[0]] = dict(zip(KEYS, values), epoch=epoch, source=parts[1], **extra)
    return out


def doc_numbers(name):
    text = (ROOT / 'docs' / DOCS[name]).read_text(encoding='utf-8')
    return text, [float(x) for x in NUM.findall(text)]


def main():
    table, agents = read_table(), agents_rows()
    summary = (ROOT / 'docs' / 'Seal全监督基线评估汇总.md').read_text(encoding='utf-8')
    problems = []
    if set(table) != set(agents):
        problems.append(f'method set mismatch: table={sorted(table)} agents={sorted(agents)}')

    for name, row in table.items():
        a = agents.get(name)
        if a is None:
            problems.append(f'{name}: missing in AGENTS.md')
            continue
        for key in KEYS:
            if abs(row[key] - a[key]) > 5.01e-6:
                problems.append(f'{name}.{key}: table={row[key]:.10g} vs AGENTS={a[key]:.10g}')
        for key in EXTRA:
            if key in a and row.get(key) is not None and abs(row[key] - a[key]) > 0.02:
                problems.append(f'{name}.{key}: table={row[key]} vs AGENTS={a[key]}')
        text, nums = doc_numbers(name)
        for key in KEYS:
            v = row[key]
            tol = max(abs(v) * 1e-4, 1e-12)
            if not any(abs(x - v) <= tol for x in nums):
                problems.append(f'{name}.{key}: table={v:.10g} not found in {DOCS[name]}')
        if str(a['epoch']) not in text:
            problems.append(f'{name}: val epoch {a["epoch"]} not in {DOCS[name]}')
        # 汇总 uses 5-decimal rendering for S/Fw/E/MAE
        for key in ('S', 'Fw', 'E', 'MAE'):
            if f'{row[key]:.5f}' not in summary and f'{row[key]:.8f}' not in summary:
                problems.append(f'{name}.{key}={row[key]:.8f} not in Seal全监督基线评估汇总.md')
        conf = row.get('source')
    print(f'{len(problems)} problem(s) over {len(table)} rows')
    for p in problems:
        print(' -', p)


if __name__ == '__main__':
    main()
