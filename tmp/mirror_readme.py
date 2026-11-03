from pathlib import Path
root = Path(".")
agents = (root / "AGENTS.md").read_bytes()
header = ("<!-- 本文件是 AGENTS.md 的镜像，便于在 GitHub 首页展示；协议与数值以 AGENTS.md 为准。 -->\n"
          "> 本文件与 [AGENTS.md](AGENTS.md) 内容一致，仅为 GitHub 首页展示而生成。\n\n").encode("utf-8")
(root / "README.md").write_bytes(header + agents)
print("README bytes:", len(header) + len(agents))
