import openpyxl
for path in (r"docs/Seal量化SOTA表.xlsx", r"tmp/backup/Seal量化SOTA表.with_ours_rows.xlsx"):
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    print("==", path, "rows", ws.max_row, "cols", ws.max_column)
    for r in (1, 2):
        print(" row", r, [(c, ws.cell(r, c).value) for c in range(1, ws.max_column + 1) if ws.cell(r, c).value])
