"""Extract the Related Work section text from a PDF for style study."""
import os
import re
import sys

import fitz

path = sys.argv[1]
start_pat = sys.argv[2] if len(sys.argv) > 2 else r'(?i)^\s*(?:\d+\s*)?(?:\.\s*)?related\s+work'
maxchars = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

doc = fitz.open(path)
text = '\n'.join(page.get_text() for page in doc)
doc.close()
lines = [ln.rstrip() for ln in text.splitlines()]
start = None
for i, ln in enumerate(lines):
    if re.match(start_pat, ln.strip()):
        start = i
        break
if start is None:
    print('NO MATCH for', start_pat, 'in', path)
    print('\n'.join(lines[:40]))
    raise SystemExit(0)
out = []
for ln in lines[start:]:
    if out and re.match(r'(?i)^\s*(?:\d+\s*)?(?:\.\s*)?(method|methodology|proposed|our (?:method|approach)|experiment|conclusion|implementation)', ln.strip()) and len(out) > 20:
        break
    out.append(ln)
body = '\n'.join(out)
print(f'=== {os.path.basename(path)} :: {len(body)} chars from line {start}')
print(body[:maxchars])
