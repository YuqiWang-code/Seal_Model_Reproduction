from pathlib import Path
p = Path(".gitignore")
s = p.read_text(encoding="utf-8")
s = s.replace("__pycache__/*.py[cod]", "__pycache__/\n*.py[cod]")
p.write_text(s, encoding="utf-8")
print("patched .gitignore")
