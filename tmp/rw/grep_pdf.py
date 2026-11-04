"""Grep a PDF's text for a keyword and print matching lines with context."""
import sys

import fitz

path = sys.argv[1]
kws = sys.argv[2:]
doc = fitz.open(path)
text = '\n'.join(page.get_text() for page in doc)
doc.close()
lines = [ln.strip() for ln in text.splitlines()]
for kw in kws:
    print(f'--- {kw}')
    for i, ln in enumerate(lines):
        if kw.lower() in ln.lower():
            ctx = ' / '.join(lines[max(0, i - 2):i + 3])
            print(f'  L{i}: {ctx[:300]}')
