from pathlib import Path
p = Path(".gitignore")
s = p.read_text(encoding="utf-8")
add = ("*.pt\n"
       "*.downloading\n"
       "Ours/weights/\n")
if "*.pt" not in s:
    s = s.replace("*.pkl\n", "*.pkl\n" + add)
    p.write_text(s, encoding="utf-8")
    print("updated .gitignore")
else:
    print("already present")
