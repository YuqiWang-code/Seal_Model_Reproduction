import ast
import json
from pathlib import Path

root=Path('/share_datasets/Seal_Model_Reproduction')
folder=root/'results/FCT-SAM-WSCOD/diagnosis_20261103'
original=root/'results/FCT-SAM-WSCOD/original_functions_20261103'
provenance=json.loads((original/'provenance.json').read_text())
names=['s_measure_folder','weighted_f_measure_folder','e_measure_folder','mae_folder']
functions=[]
for path in [root/'Ours/evaluate.py',root/'evaluate.py',original/'evaluate_entry.py']:
    text=path.read_text(encoding='utf-8-sig')
    functions.append({n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names})
assert functions[0]==functions[1]==functions[2]
report={'original_function_source':str(root/'Ours/evaluate.py'),'original_functions_unchanged':True,
    'small_target_sources':[str(root/'metrics.py'),str(root/'evaluate_small_targets.py')],
    'reinferred_seal':False,'retrained':False,'experiments':[]}
for item,small_name in zip(provenance['experiments'],['cross_domain_small_targets.json','adaptation_small_targets.json']):
    four=json.loads(Path(item['full_precision_json']).read_text())['Seal']
    small=json.loads((folder/small_name).read_text())
    assert Path(item['prediction_dir'])==Path(small['prediction_dir'])
    assert small['counts']['n_images']==item['prediction_count']==1983
    report['experiments'].append({'name':item['name'],'checkpoint':item['checkpoint'],
        'checkpoint_epoch':item['checkpoint_epoch'],'prediction_dir':item['prediction_dir'],
        'n_images':1983,'four_metric_json':item['full_precision_json'],'small_metric_json':str(folder/small_name),
        'metrics':{'S':four['S'],'Fw':four['F'],'E':four['E'],'MAE':four['MAE'],**small['metrics']}})
(folder/'eight_metric_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
