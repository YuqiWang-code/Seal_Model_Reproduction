"""检查 AS4000.tex 里 tab:comparison 表格的结构：每个逻辑行（到 \\\\ 为止）是否 8 个 &。"""

import re
from pathlib import Path

TEX = Path("Ours/RGB_T_STD_RS/AS4000.tex")
text = TEX.read_text(encoding="utf-8")
start = text.index("\\begin{table*}[t]")
end = text.index("\\end{table*}")
block = text[start:end]

# 去掉表头（到第一个 \midrule 之后），只检查数据行
body = block.split("\\midrule", 1)[1]
logical = [seg.strip() for seg in body.split("\\\\") if seg.strip()]
print(f"逻辑行数（含 \u005cmidrule 与 \u005cbottomrule 段）: {len(logical)}")
for i, seg in enumerate(logical):
    n = seg.count("&")
    head = " ".join(seg.split())[:70]
    flag = "OK " if n == 8 else ("-- " if n == 0 else "BAD")
    print(f"{flag} &={n:2d}  {head}")
