from pathlib import Path
p = Path(".gitignore")
lines = p.read_text(encoding="utf-8").splitlines()
if "*.pt" not in lines:
    i = lines.index("*.pkl") + 1
    lines[i:i] = ["*.pt", "*.downloading", "Ours/weights/"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("added *.pt / *.downloading / Ours/weights/")
else:
    print("already present")
