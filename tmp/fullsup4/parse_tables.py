import re
import pathlib

for rel in ['RPCANet-main/README.md', 'DRPCA-Net-main/README.md']:
    p = pathlib.Path(rel)
    text = p.read_text(encoding='utf-8')
    print('=' * 20, rel)
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        low = line.lower()
        if 'rpcanet' in low or 'drpca' in low or 'drpca-net' in low:
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # keep first cell + last three cells (Params / Time or Params)
            name = re.sub(r'<[^>]+>', '', cells[0])[:40]
            tail = cells[-3:]
            print(f'{name!r:45} ... tail={tail}')
