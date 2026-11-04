import openpyxl

p = r'F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\docs\Seal量化SOTA表.xlsx'
wb = openpyxl.load_workbook(p)
ws = wb[wb.sheetnames[0]]
for addr in ('A2', 'E2', 'M2', 'O2', 'E3', 'M3', 'E21', 'M21', 'O21', 'E20'):
    c = ws[addr]
    print(addr, 'numfmt=', c.number_format, 'font=', c.font.name, c.font.sz, c.font.b, 'fill=', c.fill.fgColor.rgb, 'align=', c.alignment.horizontal, c.alignment.vertical)
print('merged', [str(m) for m in ws.merged_cells.ranges][:20])
print('col widths', {k: v.width for k, v in ws.column_dimensions.items()})
print('freeze', ws.freeze_panes)
