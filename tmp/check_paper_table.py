"""校验 tab:comparison 的 \\best / \\second 标记是否与数值一致（只在基线之间比较，Ours 行不参与）。

方向：Params↓ GFLOPs↓ Fa↓ 越小越好；FPS↑ IoU↑ F1↑ Pd↑ 越大越好。
"""

import re
from pathlib import Path

TEX = Path("Ours/RGB_T_STD_RS/AS4000.tex")
text = TEX.read_text(encoding="utf-8")
block = text[text.index("\\begin{table*}[t]") : text.index("\\end{table*}")]
body = block.split("\\midrule", 1)[1]

COLS = ["Params", "GFLOPs", "FPS", "IoU", "F1", "Pd", "Fa"]
LOWER_BETTER = {"Params", "GFLOPs", "Fa"}

rows = []
for seg in body.split("\\\\"):
    seg = seg.strip()
    if not seg or "\\bottomrule" in seg:
        continue
    seg = seg.replace("\\midrule", "").strip()
    cells = [c.strip() for c in seg.split("&")]
    if len(cells) != 9:
        raise SystemExit(f"列数异常 {len(cells)}: {seg[:60]}")
    method = cells[0]
    marked = {}
    numbers = []
    for name, cell in zip(COLS, cells[2:]):
        tag = None
        m = re.fullmatch(r"\\(best|second)\{([^}]*)\}", cell)
        if m:
            tag, cell = m.group(1), m.group(2)
        marked[name] = tag
        numbers.append(float(cell))
    rows.append((method, marked, numbers))

base = [r for r in rows if r[0] != "Ours"]
print(f"{len(rows)} 行（基线 {len(base)}，Ours {len(rows) - len(base)}）\n")

ok = True
for idx, name in enumerate(COLS):
    vals = [(r[0], r[2][idx], r[1][name]) for r in base]
    ordered = sorted(vals, key=lambda t: t[1], reverse=name not in LOWER_BETTER)
    want = {"best": ordered[0][0], "second": ordered[1][0]}
    got = {t[2]: t[0] for t in vals if t[2]}
    status = "OK " if got == want else "BAD"
    if got != want:
        ok = False
    ranking = " ".join(f"{n}={v:g}" for n, v, _ in ordered[:4])
    print(f"{status} {name:7s} 应有 best={want['best']} second={want['second']} | 实际 {got} | 前四: {ranking}")

print("\n全部一致" if ok else "\n存在不一致")
