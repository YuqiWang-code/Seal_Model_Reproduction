"""Check relative Markdown links (file + #anchor) in the repository docs."""

import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
PUNCT = re.compile(r'[：:，,。.、（）()【】\[\]`\'"“”‘’!?！？*_/\\|<>~]')


def slug(heading: str) -> str:
    text = unicodedata.normalize('NFKC', heading.strip().lower())
    text = PUNCT.sub('', text)
    return text.replace(' ', '-')


def anchors(path: Path):
    found = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
        if m:
            found.add(slug(m.group(2)))
    return found


files = sorted([ROOT / 'AGENTS.md', ROOT / 'README.md'] + list((ROOT / 'docs').glob('*.md')))
cache = {}
problems = []
for md in files:
    for target in LINK.findall(md.read_text(encoding='utf-8')):
        if target.startswith(('http://', 'https://', 'mailto:')):
            continue
        path_part, _, anchor = unquote(target).partition('#')
        resolved = (md.parent / path_part).resolve() if path_part else md
        if not resolved.exists():
            problems.append(f'{md.relative_to(ROOT)}: missing file {target}')
            continue
        if anchor and resolved.suffix == '.md':
            if resolved not in cache:
                cache[resolved] = anchors(resolved)
            if anchor not in cache[resolved]:
                problems.append(f'{md.relative_to(ROOT)}: missing anchor {target}')

print(f'{len(problems)} problem(s) over {len(files)} files')
for p in problems:
    print(' -', p)
