import openpyxl
old = openpyxl.load_workbook(r"tmp/backup/Seal量化SOTA表.before_bisf_res.xlsx", data_only=True)
new = openpyxl.load_workbook(r"docs/Seal量化SOTA表.xlsx", data_only=True)
diffs = []
for name in new.sheetnames:
    a, b = old[name], new[name]
    for r in range(1, max(a.max_row, b.max_row) + 1):
        for c in range(1, max(a.max_column, b.max_column) + 1):
            if a.cell(r, c).value != b.cell(r, c).value:
                diffs.append((name, r, a.cell(r, c).column_letter, a.cell(r, c).value, b.cell(r, c).value))
print("diffs:", len(diffs))
for d in diffs:
    print(d)
ws = new["Ablation"]
print("formats row15:", [(c, ws.cell(15, c).number_format) for c in range(2, 11)])
