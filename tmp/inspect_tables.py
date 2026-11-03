import openpyxl, sys
for p in [r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\.artifact_work\Seal量化SOTA表.updated.xlsx",
          r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\ness_cross_domain\final.xlsx",
          r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\ness_cross_domain\original.xlsx",
          r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\ness_cross_domain\authored.xlsx"]:
    try:
        wb = openpyxl.load_workbook(p, data_only=True)
    except Exception as e:
        print(p, "ERR", e); continue
    ws = wb.worksheets[0]
    print("=====", p.split("\\")[-1], "rows", ws.max_row)
    for r in range(13, min(ws.max_row, 21)+1):
        print(r, [ws.cell(r, c).value for c in (1, 5, 6, 7, 8, 9, 10, 11, 12, 15)])
