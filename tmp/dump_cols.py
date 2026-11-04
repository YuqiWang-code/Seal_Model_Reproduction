import openpyxl
ws = openpyxl.load_workbook(r"docs/Seal量化SOTA表.xlsx", data_only=True).worksheets[0]
for r in (13, 20, 21, 32):
    print(r, [(c, ws.cell(r, c).value) for c in (1, 13, 14, 15, 16, 17, 18)])
