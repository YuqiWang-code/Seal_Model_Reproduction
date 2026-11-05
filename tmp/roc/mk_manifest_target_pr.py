#!/usr/bin/env python3
"""Manifest for tools/plot_pr_object.py: local copies of the object-level PR JSONs."""
import json
import os

ORDER = ['DNA-Net', 'UIU-Net', 'MSHNet', 'SCTransNet', 'RPCANet', 'HDNet', 'ILNet',
         'L2SKNet', 'BGM', 'DRPCA-Net', 'RPCANet++', 'DURD', 'SAM3', 'Ours']
STEM = {'DNA-Net': 'DNANet', 'UIU-Net': 'UIU-Net', 'MSHNet': 'MSHNet',
        'SCTransNet': 'SCTransNet', 'RPCANet': 'RPCANet', 'HDNet': 'HDNet',
        'ILNet': 'ILNet', 'L2SKNet': 'L2SKNet', 'BGM': 'BGM', 'DRPCA-Net': 'DRPCANet',
        'RPCANet++': 'RPCANetpp', 'DURD': 'DURD', 'SAM3': 'SAM3', 'Ours': 'Ours1105'}

items = []
for label in ORDER:
    path = os.path.join('tmp/roc/target_pr', STEM[label] + '.json')
    if not os.path.exists(path):
        print('missing', path)
        continue
    items.append({'label': label, 'pr': path})

json.dump({'methods': items}, open('tmp/roc/manifest_target_pr.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(len(items), [i['label'] for i in items])
