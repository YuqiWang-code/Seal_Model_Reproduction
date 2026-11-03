#!/usr/bin/env python3
"""Generate README.md as a GitHub-facing mirror of AGENTS.md."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
agents = (root / 'AGENTS.md').read_text(encoding='utf-8')
intro = ('<!-- 本文件是 AGENTS.md 的镜像，便于在 GitHub 首页展示；协议与数值以 AGENTS.md 为准。 -->\n'
         '> 本文件与 [AGENTS.md](AGENTS.md) 内容一致，仅为 GitHub 首页展示而生成。\n\n')
(root / 'README.md').write_text(intro + agents, encoding='utf-8')
print('README.md written', len(intro + agents), 'chars')
