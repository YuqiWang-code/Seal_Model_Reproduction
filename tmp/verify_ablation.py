import openpyxl
p = r"docs/Seal量化SOTA表.xlsx"
wb = openpyxl.load_workbook(p)
ws = wb["Ablation"]
print("formats row14:", [(c, ws.cell(14, c).number_format) for c in range(2, 11)])
print("formats row15:", [(c, ws.cell(15, c).number_format) for c in range(2, 11)])
old = openpyxl.load_workbook(r"tmp/backup/Seal量化SOTA表.before_bisf_row.xlsx", data_only=True)
new = openpyxl.load_workbook(p, data_only=True)
print("sheets:", new.sheetnames)
diffs = []
for name in new.sheetnames:
    a, b = old[name], new[name]
    for r in range(1, max(a.max_row, b.max_row) + 1):
        for c in range(1, max(a.max_column, b.max_column) + 1):
            if a.cell(r, c).value != b.cell(r, c).value:
                diffs.append((name, r, a.cell(r, c).column_letter, a.cell(r, c).value, b.cell(r, c).value))
print("diffs:", len(diffs))
for d in diffs[:12]:
    print(d)
