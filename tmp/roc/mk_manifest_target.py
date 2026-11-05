import json
import os

root = 'tmp/roc/target'
# (file stem, display label) sorted by plateau Pd (highest first) = 表内 Pd 排序
pairs = [
    ('Ours', 'Ours'),
    ('HDNet', 'HDNet'),
    ('MSHNet', 'MSHNet'),
    ('RPCANet', 'RPCANet'),
    ('DNANet', 'DNA-Net'),
    ('BGM', 'BGM'),
    ('L2SKNet', 'L2SKNet'),
    ('DURD', 'DURD'),
    ('RPCANetpp', 'RPCANet++'),
    ('DRPCANet', 'DRPCA-Net'),
    ('UIU-Net', 'UIU-Net'),
    ('SCTransNet', 'SCTransNet'),
    ('ILNet', 'ILNet'),
]
m = {'methods': [{'label': lab, 'json': os.path.join(root, stem + '.json')} for stem, lab in pairs]}
json.dump(m, open('tmp/roc/manifest_target_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(m['methods']))
