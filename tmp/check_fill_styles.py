import openpyxl
ws = openpyxl.load_workbook(r"tmp/backup/table_fill_test.xlsx").worksheets[0]
for r in (32, 33, 34, 35):
    print(r, [(ws.cell(r, c).column_letter, ws.cell(r, c).number_format,
               ws.cell(r, c).font.size, ws.cell(r, c).alignment.horizontal,
               ws.cell(r, c).border.left.style, ws.cell(r, c).fill.fgColor.rgb) for c in (1, 5, 12)])
