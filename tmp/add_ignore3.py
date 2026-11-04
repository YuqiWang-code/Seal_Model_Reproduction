from pathlib import Path
p = Path(".gitignore")
lines = p.read_text(encoding="utf-8").splitlines()
if "tmp/1104_test_maps/" not in lines:
    lines += ["", "# 解压出的临时预测副本（原 zip 已入库）", "tmp/1104_test_maps/"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("added tmp/1104_test_maps/")
else:
    print("already present")
