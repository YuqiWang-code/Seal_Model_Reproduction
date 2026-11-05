#!/usr/bin/env python3
"""Consistency check: object-level recall at t=128 must equal the table Pd."""
import json
import os

TABLE = {  # Pd @ t=128 from docs/Seal量化SOTA表.xlsx rows 21-32 (AGENTS table)
    'DNA-Net': 0.79046, 'MSHNet': 0.80943, 'UIU-Net': 0.68739,
    'SCTransNet': 0.62210, 'RPCANet': 0.80618, 'HDNet': 0.82135,
    'ILNet': 0.58946, 'RPCANet++': 0.76920, 'L2SKNet': 0.77204,
    'BGM': 0.78681, 'DRPCA-Net': 0.76852, 'DURD': 0.77584,
    'SAM3': None, 'Ours': 0.89151,
}
STEM = {'DNA-Net': 'DNANet', 'UIU-Net': 'UIU-Net', 'MSHNet': 'MSHNet',
        'SCTransNet': 'SCTransNet', 'RPCANet': 'RPCANet', 'HDNet': 'HDNet',
        'ILNet': 'ILNet', 'L2SKNet': 'L2SKNet', 'BGM': 'BGM', 'DRPCA-Net': 'DRPCANet',
        'RPCANet++': 'RPCANetpp', 'DURD': 'DURD', 'SAM3': 'SAM3', 'Ours': 'Ours1105'}

for label, stem in STEM.items():
    path = f'tmp/roc/target_pr/{stem}.json'
    if not os.path.exists(path):
        continue
    rec = json.load(open(path, encoding='utf-8'))
    ths = rec['thresholds']
    j = ths.index(128) if 128 in ths else min(range(len(ths)), key=lambda k: abs(ths[k] - 128))
    r128 = rec['recall'][j]
    want = TABLE.get(label)
    delta = '' if want is None else f'  table={want:.5f}  diff={r128 - want:+.2e}'
    print(f'{label:11s} t={ths[j]:3d} recall={r128:.5f} precision={rec["precision"][j]:.5f}{delta}')
