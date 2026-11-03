import openpyxl
ws = openpyxl.load_workbook(r"docs\Seal量化SOTA表.xlsx", data_only=True).worksheets[0]
print("rows", ws.max_row, "cols", ws.max_column)
for r in range(17, 29):
    row = [ws.cell(r, c).value for c in range(1, 13)]
    print(r, row[0], "|", row[2], "|", row[3], "|", " ".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in row[4:12]))
