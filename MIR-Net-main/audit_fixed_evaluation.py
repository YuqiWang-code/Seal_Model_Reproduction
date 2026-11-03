"""Check evaluator source and evaluate complete prediction folders."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

FUNCTIONS = ['s_measure_folder', 'weighted_f_measure_folder', 'e_measure_folder', 'mae_folder']

def functions(path):
    source = path.read_text(encoding='utf-8-sig')
    return {node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
            and node.name in FUNCTIONS}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pred', type=Path)
    parser.add_argument('--gt_dir', type=Path)
    args = parser.parse_args()
    assert functions(args.root / 'evaluate.py') == functions(args.reference)
    spec = importlib.util.spec_from_file_location('original_metrics', args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gt = args.gt_dir or args.root / 'Seal_data/test/GT'
    paths = [args.pred] if args.pred else [args.root / 'results/MIR-Net/Seal' / name for name in
        ['published_test_maps', 'finetuned_test_maps', 'box_sam_test_maps', 'box_sam_150_test_maps']]
    result = {}
    for pred in paths:
        expected = {p.name for p in gt.iterdir() if p.is_file()}
        actual = {p.name for p in pred.iterdir() if p.is_file()}
        assert expected == actual and len(expected) == 1983, (pred, len(actual))
        metrics = {name: float(getattr(module, fn)(str(pred), str(gt))[key])
                   for name, fn, key in zip(['S','Fw','E','MAE'], FUNCTIONS,
                                            ['mean_S','mean_Fw','mean_E','mean_MAE'])}
        result[pred.name] = dict(prediction_count=len(actual), gt_dir=str(gt), metrics=metrics)
        print(pred.name, metrics, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
