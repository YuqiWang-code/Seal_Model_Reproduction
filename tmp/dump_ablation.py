import openpyxl
wb = openpyxl.load_workbook(r"docs/Seal量化SOTA表.xlsx", data_only=True)
ws = wb["Ablation"]
print("rows", ws.max_row, "cols", ws.max_column)
for r in range(1, ws.max_row + 1):
    print(r, [ws.cell(r, c).value for c in range(1, ws.max_column + 1)])
