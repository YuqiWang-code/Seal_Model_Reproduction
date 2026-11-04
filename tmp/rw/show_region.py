"""Print the region of a PDF's text around a keyword (case-insensitive)."""
import sys

import fitz

path = sys.argv[1]
kw = sys.argv[2]
maxchars = int(sys.argv[3]) if len(sys.argv) > 3 else 7000

doc = fitz.open(path)
text = '\n'.join(page.get_text() for page in doc)
doc.close()
low = text.lower()
i = low.find(kw.lower())
if i < 0:
    print('NOT FOUND:', kw)
    raise SystemExit(0)
start = text.rfind('\n', 0, i)
print(text[start:start + maxchars])
