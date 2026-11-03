"""Confirm the fixed public-checkpoint Seal result and retained experiment state."""
import ast
import json
from datetime import datetime
from pathlib import Path

root = Path('/share_datasets/Seal_Model_Reproduction')
diagnosis = root / 'results/FCT-SAM-WSCOD/diagnosis_20261103'
original = root / 'results/FCT-SAM-WSCOD/original_functions_20261103'
names = {'s_measure_folder', 'weighted_f_measure_folder', 'e_measure_folder', 'mae_folder'}
def functions(path):
    text = path.read_text(encoding='utf-8-sig')
    return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name in names}
reference = functions(root / 'Ours/evaluate.py')
assert len(reference) == 4
assert functions(root / 'evaluate.py') == reference
assert functions(original / 'evaluate_entry.py') == reference
combined = json.loads((diagnosis / 'eight_metric_results.json').read_text())
experiment = next(e for e in combined['experiments'] if e['name'] == 'cross_domain')
pred = Path(experiment['prediction_dir'])
gt = root / 'Seal_data/test/GT'
pred_names = {p.name for p in pred.glob('*.png')}
gt_names = {p.name for p in gt.glob('*.png')}
assert pred_names == gt_names and len(pred_names) == 1983
small = json.loads(Path(experiment['small_metric_json']).read_text())
four = json.loads(Path(experiment['four_metric_json']).read_text())['Seal']
assert small['prediction_dir'] == str(pred) and small['gt_dir'] == str(gt)
assert small['counts']['n_images'] == 1983
assert experiment['metrics'] == {'S': four['S'], 'Fw': four['F'], 'E': four['E'],
                                 'MAE': four['MAE'], **small['metrics']}
assert Path(experiment['checkpoint']).is_file()
box = root / 'results/FCT-SAM-WSCOD/Seal_box_sam'
record = {
    'checked_at': datetime.now().astimezone().isoformat(),
    'protocol': 'public full COD checkpoint -> Seal test; no Seal training or model selection',
    'checkpoint': experiment['checkpoint'], 'checkpoint_epoch': experiment['checkpoint_epoch'],
    'author_explicitly_designated_final_checkpoint': 'not confirmed by available release materials',
    'prediction_dir': str(pred), 'gt_dir': str(gt), 'n_images': len(pred_names),
    'missing': [], 'extra': [], 'original_four_functions_match': True,
    'evaluation_snapshot_matches': True, 'metrics': experiment['metrics'],
    'four_metric_json': experiment['four_metric_json'], 'small_metric_json': experiment['small_metric_json'],
    'reinferred': False, 'retrained': False,
    'historical_box_adaptation': {'labels_retained': len(list((box / 'labels').glob('*.png'))),
        'training_started': (box / 'history.jsonl').exists() or (box / 'last.pth').exists(),
        'status': 'cancelled; no restart'},
}
(diagnosis / 'cross_domain_protocol_check.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
