import ast
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

root = Path('/share_datasets/Seal_Model_Reproduction')
destination = root / 'results/FCT-SAM-WSCOD/original_functions_20261103'
destination.mkdir(parents=True, exist_ok=True)
names = ['s_measure_folder', 'weighted_f_measure_folder', 'e_measure_folder', 'mae_folder']
def functions(path):
    source = path.read_text(encoding='utf-8-sig')
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in names}
same = {name: functions(root / 'evaluate.py')[name] == functions(root / 'Ours/evaluate.py')[name] for name in names}
assert all(same.values()), same
shutil.copyfile(root / 'evaluate.py', destination / 'evaluate_entry.py')
shutil.copyfile(root / 'Ours/evaluate.py', destination / 'ours_evaluate_source.py')
gt = root / 'Seal_data/test/GT'
expected = {path.stem for path in gt.iterdir() if path.is_file()}
experiments = [
    ('cross_domain', 'results/FCT-SAM-WSCOD/Seal/test_maps', 'FCT-SAM-WSCOD-main/pretrain/Pvtv2_TE_new_dataset_batch4_epoch60.pth'),
    ('seal_adaptation', 'results/FCT-SAM-WSCOD/Seal_retrain_768_1fg5bg/test_maps', 'results/FCT-SAM-WSCOD/Seal_retrain_768_1fg5bg/best.pth'),
]
metadata = {'evaluated_at': datetime.now().astimezone().isoformat(), 'function_source': str(root / 'Ours/evaluate.py'),
            'function_source_matches': same, 'gt_dir': str(gt), 'reinferred': False, 'retrained': False, 'experiments': []}
for label, prediction, checkpoint in experiments:
    folder = root / prediction
    files = list(folder.iterdir())
    actual = {path.stem for path in files if path.is_file()}
    filenames = {path.name for path in files if path.is_file()}
    gt_filenames = {path.name for path in gt.iterdir() if path.is_file()}
    assert len(actual) == len(files) == len(expected) == 1983 and actual == expected
    assert filenames == gt_filenames
    saved = torch.load(root / checkpoint, map_location='cpu', weights_only=False)
    if label == 'seal_adaptation':
        assert saved['epoch'] == 45 and saved['input_size'] == 768
    output = destination / f'{label}_metrics_full.json'
    log = destination / f'{label}_evaluate.log'
    assert not output.exists() and not log.exists(), 'Existing evaluation artifacts must be retained'
    command = [sys.executable, str(root / 'evaluate.py'), '--pred_root', str(folder), '--gt_dir', str(gt), '--json_output', str(output)]
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    metadata['experiments'].append({'name': label, 'prediction_dir': str(folder), 'checkpoint': str(root / checkpoint),
        'checkpoint_epoch': saved['epoch'], 'input_size': saved.get('input_size', 384), 'prediction_count': len(files),
        'gt_count': len(expected), 'missing': [], 'extra': [], 'filenames_match': True, 'command': command,
        'log': str(log), 'full_precision_json': str(output), 'pid': process.pid})
(destination / 'provenance.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(metadata, indent=2, ensure_ascii=False))
