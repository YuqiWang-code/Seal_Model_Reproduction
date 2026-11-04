import json
import os
import subprocess

ROOT = '/share_datasets/Seal_Model_Reproduction'
PY = '/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python'
SCRIPT = os.path.join(ROOT, 'tmp', 'fullsup4', 'cmp_params.py')

DIRS = {
    'dnanet': 'DNANet-main', 'mshnet': 'MSHNet-main', 'uiunet': 'UIU-Net-main',
    'sctransnet': 'SCTransNet-main', 'rpcanet': 'RPCANet-main', 'hdnet': 'HDNet-main',
    'ilnet': 'ILNet-main', 'rpcanetpp': 'RPCANet-main', 'l2sknet': 'L2SKNet-main',
    'bgm': 'BGM-main', 'drpcanet': 'DRPCA-Net-main', 'durd': 'DURD-main',
}
only = os.environ.get('KINDS')
kinds = only.split(',') if only else list(DIRS)
for kind in kinds:
    p = subprocess.run([PY, SCRIPT, kind], cwd=os.path.join(ROOT, DIRS[kind]),
                       capture_output=True, text=True)
    out = (p.stdout.strip().splitlines() or [''])[-1]
    if p.returncode != 0:
        out = f'FAIL {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""}'
    print(out, flush=True)
