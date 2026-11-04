import openpyxl
wb = openpyxl.load_workbook(r"docs/Seal量化SOTA表.xlsx", data_only=True)
print("sheets:", wb.sheetnames)
for name in wb.sheetnames:
    ws = wb[name]
    print("==", name, "rows", ws.max_row, "cols", ws.max_column)
