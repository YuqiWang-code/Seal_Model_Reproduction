"""Fetch abstracts via Semantic Scholar (fallback: OpenAlex with mailto) for the 8 methods."""
import json
import sys
import time
import urllib.parse
import urllib.request

TITLES = {
    'DNA-Net': 'Dense Nested Attention Network for Infrared Small Target Detection',
    'MSHNet': 'Infrared Small Target Detection with Scale and Location Sensitivity',
    'HDNet': 'A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection',
    'L2SKNet': 'Saliency at the Helm Steering Infrared Small Target Detection With Learnable Kernels',
    'BGM': 'Forgetting the Background A Masking Approach for Enhanced Infrared Small Target Detection',
    'RPCANet': 'RPCANet Deep Unfolding RPCA Based Infrared Small Target Detection',
    'RPCANet++': 'RPCANet++ Deep Interpretable Robust PCA for Sparse Object Segmentation',
    'SAM3': 'SAM 3 Segment Anything with Concepts',
}


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research-check'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))


def s2(title):
    url = ('https://api.semanticscholar.org/graph/v1/paper/search?query=' +
           urllib.parse.quote(title) +
           '&limit=2&fields=title,abstract,venue,year,authors,externalIds')
    return get_json(url).get('data', [])


only = sys.argv[1:] or list(TITLES)
for key in only:
    print(f'\n### {key}')
    try:
        hits = s2(TITLES[key])
    except Exception as exc:  # noqa: BLE001
        print('  S2 FAIL', exc)
        continue
    for h in hits:
        auths = [a.get('name') for a in (h.get('authors') or [])][:4]
        print(f"  - {h.get('year')} | {h.get('venue')} | {auths} | {h.get('externalIds')}")
        print(f"    T: {h.get('title')}")
        ab = (h.get('abstract') or '(no abstract)').replace('\n', ' ')
        print(f"    A: {ab[:1600]}")
    time.sleep(3.0)
