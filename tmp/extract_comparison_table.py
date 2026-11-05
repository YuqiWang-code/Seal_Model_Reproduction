"""从 AS4000.tex 里抽出 tab:comparison 表格，另存为可单独粘贴的片段。"""

from pathlib import Path

TEX = Path("Ours/RGB_T_STD_RS/AS4000.tex")
OUT = Path("tmp/table_comparison.tex")

text = TEX.read_text(encoding="utf-8")
start = text.index("\\begin{table*}[t]")
end = text.index("\\end{table*}") + len("\\end{table*}")
OUT.write_text(text[start:end] + "\n", encoding="utf-8", newline="\n")
print(f"写入 {OUT}（{OUT.read_text(encoding='utf-8').count(chr(10))} 行）")
