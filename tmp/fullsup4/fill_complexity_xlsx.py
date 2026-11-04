"""Fill Params / GFLOPs / FPS (cols 13-15) for the 12 fully-supervised baselines.

Values come from results/complexity/<kind>.json measured on the server with
tools/measure_complexity_one.py (thop MACs @ 1x256x256, FP32, batch 1, RTX 5090,
clock-warmed, FPS = 1000 / median latency). Columns are located by header text,
never by a fixed index.
"""
import openpyxl

XLSX = r'F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\docs\Seal量化SOTA表.xlsx'

# method row label -> (params_M, gflops, fps)
DATA = {
    'DNA-Net': (4.697, 14.282, 64.22),
    'MSHNet': (4.066, 6.112, 152.32),
    'UIU-Net': (50.541, 54.501, 118.39),
    'SCTransNet': (11.326, 10.119, 109.15),
    'RPCANet': (0.680, 44.569, 211.29),
    'HDNet': (3.676, 5.697, 85.54),
    'ILNet': (4.040, 21.421, 108.39),
    'RPCANet++': (2.962, 190.669, 80.42),
    'L2SKNet': (0.899, 6.895, 332.16),
    'BGM': (4.076, 6.773, 212.14),
    'DRPCA-Net': (1.169, 73.837, 11.23),
    'DURD': (2.466, 52.847, 125.34),
}

HEADER_ROW = 2
COL_PARAMS = 'Params↓'
COL_GFLOPS = 'GFLOPs↓'
COL_FPS = 'FPS↑'

wb = openpyxl.load_workbook(XLSX)
ws = wb[wb.sheetnames[0]]

headers = {}
for col in range(1, ws.max_column + 1):
    v = ws.cell(HEADER_ROW, col).value
    if isinstance(v, str):
        headers[v.strip()] = col
missing = [h for h in (COL_PARAMS, COL_GFLOPS, COL_FPS) if h not in headers]
if missing:
    raise SystemExit(f'missing headers: {missing}; got {sorted(headers)}')
c_par, c_gf, c_fps = headers[COL_PARAMS], headers[COL_GFLOPS], headers[COL_FPS]

rows = {}
for r in range(HEADER_ROW + 1, ws.max_row + 1):
    name = ws.cell(r, 1).value
    if isinstance(name, str):
        rows[name.strip()] = r

filled = []
for method, (par, gf, fps) in DATA.items():
    if method not in rows:
        raise SystemExit(f'method not found in sheet: {method}')
    r = rows[method]
    ws.cell(r, c_par).value = par
    ws.cell(r, c_gf).value = gf
    ws.cell(r, c_fps).value = fps
    filled.append((method, r, par, gf, fps))

wb.save(XLSX)
print(f'cols: params={c_par} gflops={c_gf} fps={c_fps}')
for row in filled:
    print('  ', row)
