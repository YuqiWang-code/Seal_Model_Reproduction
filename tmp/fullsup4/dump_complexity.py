import json
import glob
import os

for f in sorted(glob.glob('/share_datasets/Seal_Model_Reproduction/results/complexity/*.json')):
    name = os.path.basename(f)[:-5]
    raw = open(f, encoding='utf-8').read()
    print('==', name, len(raw), 'bytes')
    try:
        d = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print('   PARSE FAIL', exc, repr(raw[:200]))
        continue
    print('   ', json.dumps(d, ensure_ascii=False))
