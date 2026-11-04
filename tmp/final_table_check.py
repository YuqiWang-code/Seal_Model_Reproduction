import openpyxl
ws = openpyxl.load_workbook(r"docs/Seal量化SOTA表.xlsx").worksheets[0]
print("dims", ws.max_row, ws.max_column, "| merged", sorted(str(m) for m in ws.merged_cells.ranges))
for r in (1, 2):
    print(r, [(c.column_letter, c.value) for c in ws[r] if c.value])
print("row3:", [(c.column_letter, ws.cell(3, c).value) for c in (12, 13, 14, 15, 16, 17, 18)])
