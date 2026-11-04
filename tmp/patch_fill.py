from pathlib import Path
p = Path("tmp/fill_ours_rows.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''HEADERS = {"S↑": "S", "Fw↑": "Fw", "E↑": "E", "MAE↓": "MAE",
           "IoU": "IoU", "F1": "F1", "Pd": "Pd", "Fa": "Fa"}''',
'''HEADERS = {"S↑": "S", "Fw↑": "Fw", "E↑": "E", "MAE↓": "MAE",
           "IoU": "IoU", "F1": "F1", "Pd": "Pd", "Fa": "Fa",
           "重训/测试": "note", "备注": "note"}''')

s = s.replace(
'''    missing = [k for k in ("S", "Fw", "E", "MAE", "IoU", "F1", "Pd", "Fa") if k not in columns]
    if missing:
        raise RuntimeError(f"表头缺少列: {missing}")
    return columns''',
'''    required = ("S", "Fw", "E", "MAE", "IoU", "F1", "Pd", "Fa")
    missing = [k for k in required if k not in columns]
    if missing:
        raise RuntimeError(f"表头缺少列: {missing}")
    if "note" not in columns:
        raise RuntimeError("表头里找不到『重训/测试』或『备注』列")
    return columns''')

s = s.replace('        ws.cell(row, 15).value = note', '        ws.cell(row, columns["note"]).value = note')
p.write_text(s, encoding="utf-8")
print("patched")
