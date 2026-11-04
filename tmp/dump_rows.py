import openpyxl
ws = openpyxl.load_workbook(r"tmp/backup/Seal量化SOTA表.with_ours_rows.xlsx", data_only=True).worksheets[0]
print("rows", ws.max_row, "cols", ws.max_column)
for r in (32, 33, 34, 35):
    print("row", r)
    for c in range(1, 19):
        v = ws.cell(r, c).value
        if v is not None:
            print("   ", ws.cell(r, c).column_letter, repr(v)[:90])
