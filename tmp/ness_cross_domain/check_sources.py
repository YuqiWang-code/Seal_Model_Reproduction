import ast
from pathlib import Path
names={'s_measure_folder','weighted_f_measure_folder','e_measure_folder','mae_folder'}
def functions(path):
    source=Path(path).read_text(encoding='utf-8')
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
            if isinstance(n,ast.FunctionDef) and n.name in names}
assert functions('Ours/evaluate.py')==functions('tmp/ness_cross_domain/evaluate.py')
for name in ('metrics.py','evaluate_small_targets.py'):
    assert Path(name).read_text(encoding='utf-8')==Path('tmp/ness_cross_domain',name).read_text(encoding='utf-8'),name
print('Server metric implementations match fixed local sources.')
