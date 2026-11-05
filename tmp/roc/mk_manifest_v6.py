import json
import os

root = 'tmp/roc'
server = os.path.join(root, 'server')
pairs = [
    ('DNA-Net', os.path.join(server, 'DNANet.json')),
    ('UIU-Net', os.path.join(server, 'UIU-Net.json')),
    ('MSHNet', os.path.join(server, 'MSHNet.json')),
    ('SCTransNet', os.path.join(server, 'SCTransNet.json')),
    ('RPCANet', os.path.join(server, 'RPCANet.json')),
    ('HDNet', os.path.join(server, 'HDNet.json')),
    ('ILNet', os.path.join(server, 'ILNet.json')),
    ('L2SKNet', os.path.join(server, 'L2SKNet.json')),
    ('BGM', os.path.join(server, 'BGM.json')),
    ('DRPCA-Net', os.path.join(server, 'DRPCANet.json')),
    ('RPCANet++', os.path.join(server, 'RPCANetpp.json')),
    ('DURD', os.path.join(server, 'DURD.json')),
    ('SAM3', os.path.join(root, 'SAM3__test_maps.json')),
    ('Ours', os.path.join(root, 'Ours1105__test_maps.json')),
]
m = {'methods': [{'label': lab, 'hist': p} for lab, p in pairs]}
json.dump(m, open('tmp/roc/manifest_v6.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(m['methods']), 'methods; missing:', [p for _, p in pairs if not os.path.exists(p)])
