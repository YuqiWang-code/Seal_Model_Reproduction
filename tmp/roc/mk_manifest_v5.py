import json
import os

root = 'tmp/roc'
server = os.path.join(root, 'server')
pairs = [
    ('Ours', os.path.join(root, 'Ours1105__test_maps.json')),
    ('DNA-Net', os.path.join(server, 'DNANet.json')),
    ('MSHNet', os.path.join(server, 'MSHNet.json')),
    ('RPCANet', os.path.join(server, 'RPCANet.json')),
    ('HDNet', os.path.join(server, 'HDNet.json')),
    ('L2SKNet', os.path.join(server, 'L2SKNet.json')),
    ('BGM', os.path.join(server, 'BGM.json')),
    ('RPCANet++', os.path.join(server, 'RPCANetpp.json')),
    ('SAM3', os.path.join(root, 'SAM3__test_maps.json')),
]
m = {'methods': [{'label': lab, 'hist': p} for lab, p in pairs]}
json.dump(m, open('tmp/roc/manifest_v5.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(m['methods']), 'methods; missing:', [p for _, p in pairs if not os.path.exists(p)])
