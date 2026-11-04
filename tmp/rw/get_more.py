"""Download DNA-Net paper and a few 2024-2026 IRSTD PDFs for related-work style + facts."""
import os
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdfs')
os.makedirs(OUT, exist_ok=True)

PDFS = {
    'dnanet': 'https://arxiv.org/pdf/2108.03555v2',
}
for name, url in PDFS.items():
    path = os.path.join(OUT, name + '.pdf')
    if os.path.exists(path) and os.path.getsize(path) > 50000:
        print(f'{name}: cached')
        continue
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, 'wb') as fh:
            fh.write(resp.read())
        print(f'{name}: {os.path.getsize(path) / 1e6:.1f} MB')
    except Exception as exc:  # noqa: BLE001
        print(f'{name}: FAIL {type(exc).__name__}: {exc}')
