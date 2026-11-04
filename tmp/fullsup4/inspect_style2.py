import openpyxl

p = r'F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\docs\Seal量化SOTA表.xlsx'
wb = openpyxl.load_workbook(p)
ws = wb[wb.sheetnames[0]]
for r in (21, 25, 32):
    for col in 'MN O'.replace(' ', ''):
        c = ws[f'{col}{r}']
        print(f'{col}{r}', 'numfmt=', c.number_format, 'font=', c.font.name, c.font.sz, 'border=', c.border.left.style, c.border.bottom.style, 'fill=', c.fill.fgColor.rgb)
