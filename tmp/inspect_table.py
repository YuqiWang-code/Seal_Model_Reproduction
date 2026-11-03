import openpyxl
wb = openpyxl.load_workbook(r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\docs\Seal量化SOTA表.xlsx", data_only=True)
for ws in wb.worksheets:
    print("SHEET:", ws.title, ws.max_row, ws.max_column)
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 25)):
        vals = [c.value for c in row]
        print(row[0].row, vals)
