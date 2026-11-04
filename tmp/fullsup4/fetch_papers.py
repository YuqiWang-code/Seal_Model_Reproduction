"""Download PDFs and extract text so published Params/FLOPs tables can be grepped."""
import os
import re
import sys
import urllib.request

import fitz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'papers')
os.makedirs(OUT, exist_ok=True)

PAPERS = {
    'ilnet_arxiv': 'https://arxiv.org/pdf/2309.13646v1',
    'uiunet_arxiv': 'https://arxiv.org/pdf/2301.00942',
    'rpcanet_arxiv': 'https://arxiv.org/pdf/2310.14937',
    'lpcanet_arxiv': 'https://arxiv.org/pdf/2607.04603',
    'datransnet_arxiv': 'https://arxiv.org/pdf/2409.19599v5',
}
KEYS = re.compile(r'(Param|FLOP|GMac|MACs|FPS|Runtime|Time)', re.I)

for name, url in PAPERS.items():
    txt_path = os.path.join(OUT, name + '.txt')
    if os.path.exists(txt_path):
        print(f'== {name}: cached')
    else:
        pdf_path = os.path.join(OUT, name + '.pdf')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as resp, open(pdf_path, 'wb') as fh:
                fh.write(resp.read())
            doc = fitz.open(pdf_path)
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()
            with open(txt_path, 'w', encoding='utf-8') as fh:
                fh.write(text)
            print(f'== {name}: {len(text)} chars from {len(open(pdf_path, "rb").read())} bytes')
        except Exception as exc:  # noqa: BLE001
            print(f'== {name}: FETCH FAIL {exc}')
            continue
    text = open(txt_path, encoding='utf-8').read()
    hits = [ln.strip() for ln in text.splitlines() if KEYS.search(ln)]
    print(f'   lines mentioning Params/FLOPs/FPS: {len(hits)}')
    for ln in hits[:25]:
        print('   |', ln[:160])
