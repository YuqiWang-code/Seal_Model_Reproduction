import json
import re

p = '/share_datasets/Seal_Model_Reproduction/results/complexity/sctransnet.json'
raw = open(p, encoding='utf-8').read()
i = raw.find('{')
fixed = raw[i:]
d = json.loads(fixed)
with open(p, 'w', encoding='utf-8') as fh:
    json.dump(d, fh, indent=2, ensure_ascii=False)
    fh.write('\n')
print(json.dumps(d, ensure_ascii=False))
