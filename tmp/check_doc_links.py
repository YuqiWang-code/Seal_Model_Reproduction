"""Check that relative Markdown links in docs/*.md resolve to existing files."""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')

problems = []
for md in sorted((ROOT / 'docs').glob('*.md')):
    text = md.read_text(encoding='utf-8')
    for target in LINK.findall(text):
        if target.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        path_part = unquote(target.split('#')[0])
        if not path_part:
            continue
        resolved = (md.parent / path_part).resolve()
        if not resolved.exists():
            problems.append(f'{md.name}: {target}')

print(f'{len(problems)} broken relative link(s)')
for p in problems:
    print(' -', p)
