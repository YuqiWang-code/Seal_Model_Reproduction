import openpyxl

p = r'F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\docs\Seal量化SOTA表.xlsx'
wb = openpyxl.load_workbook(p)
print('sheets', wb.sheetnames)
ws = wb[wb.sheetnames[0]]
print('dims', ws.max_row, ws.max_column)
for r in (1, 2, 3, 20, 21, 32):
    vals = []
    for c in range(1, ws.max_column + 1):
        v = ws.cell(r, c).value
        vals.append(f'{c}:{v!r}')
    print('ROW', r, ' | '.join(vals))
