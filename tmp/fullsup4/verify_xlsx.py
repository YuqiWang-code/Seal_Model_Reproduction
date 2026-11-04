import subprocess
import io
import sys

import openpyxl

REPO = r'F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction'
XLSX = REPO + r'\docs\Seal量化SOTA表.xlsx'

head = subprocess.run(
    ['git', '-C', REPO, 'show', 'HEAD:docs/Seal量化SOTA表.xlsx'],
    capture_output=True, check=True).stdout

old = openpyxl.load_workbook(io.BytesIO(head))
new = openpyxl.load_workbook(XLSX)
print('sheets old/new', old.sheetnames, new.sheetnames)
wo, wn = old[old.sheetnames[0]], new[new.sheetnames[0]]
print('dims old', wo.max_row, wo.max_column, 'new', wn.max_row, wn.max_column)
print('images old', len(wo._images), 'new', len(wn._images))
print('charts old', len(wo._charts), 'new', len(wn._charts))

diffs = 0
for r in range(1, max(wo.max_row, wn.max_row) + 1):
    for c in range(1, max(wo.max_column, wn.max_column) + 1):
        a, b = wo.cell(r, c).value, wn.cell(r, c).value
        if a != b:
            diffs += 1
            print(f'DIFF {openpyxl.utils.get_column_letter(c)}{r}: {a!r} -> {b!r}')
print('total diffs', diffs)
