"""Reevaluate saved NESS-Net predictions with the fixed original metrics."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("s_measure_folder", "weighted_f_measure_folder", "e_measure_folder", "mae_folder")


def functions(path):
    source = path.read_text()
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in NAMES}


def main():
    assert functions(ROOT / "evaluate.py") == functions(ROOT / "Ours/evaluate.py")
    spec = importlib.util.spec_from_file_location("fixed", ROOT / "Ours/evaluate.py")
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    output = ROOT / "results/NESS-Net/Seal/original_eval"
    output.mkdir(exist_ok=True)
    (output / "evaluate_source.py").write_text((ROOT / "Ours/evaluate.py").read_text())
    gt = ROOT / "Seal_data/test/GT"
    expected = {p.name for p in gt.glob("*.png")}
    assert len(expected) == 1983
    rows = {}
    for name in ("refinement_namlab", "general_namlab", "general_canny", "refinement_canny"):
        predictions = output.parent / f"{name}_test_maps"
        assert {p.name for p in predictions.glob("*.png")} == expected, name
        metrics = {}
        for function, key, mean in zip(NAMES, ("S", "F", "E", "MAE"), ("mean_S", "mean_Fw", "mean_E", "mean_MAE")):
            metrics[key] = float(getattr(fixed, function)(str(predictions), str(gt))[mean])
        rows[name] = {"predictions": str(predictions), "count": len(expected), **metrics}
        (output / "published_metrics.json").write_text(json.dumps(rows, indent=2))
        print(name, metrics, flush=True)


if __name__ == "__main__":
    main()
