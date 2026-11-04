"""Download reference PDFs for related-work style study."""
import os
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdfs')
os.makedirs(OUT, exist_ok=True)

PDFS = {
    'mshnet_cvpr24': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.pdf',
    'rpcanet_wacv24': 'https://openaccess.thecvf.com/content/WACV2024/papers/Wu_RPCANet_Deep_Unfolding_RPCA_Based_Infrared_Small_Target_Detection_WACV_2024_paper.pdf',
    'lpcanet': 'https://arxiv.org/pdf/2607.04603v3',
    'sam3': 'https://arxiv.org/pdf/2511.16719',
}

for name, url in PDFS.items():
    path = os.path.join(OUT, name + '.pdf')
    if os.path.exists(path) and os.path.getsize(path) > 50000:
        print(f'{name}: cached {os.path.getsize(path) / 1e6:.1f} MB')
        continue
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, 'wb') as fh:
            fh.write(resp.read())
        print(f'{name}: {os.path.getsize(path) / 1e6:.1f} MB')
    except Exception as exc:  # noqa: BLE001
        print(f'{name}: FAIL {type(exc).__name__}: {exc}')
