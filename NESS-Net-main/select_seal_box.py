"""Select a Seal box adaptation checkpoint on validation Fw, then test once."""

import argparse
import json
import subprocess
import sys
import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_evaluate", ROOT.parent / "Ours" / "evaluate.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def complete(directory, gt_dir):
    expected = {p.name for p in gt_dir.glob("*.png")}
    return directory.is_dir() and {p.name for p in directory.glob("*.png")} == expected


def evaluate(directory, gt_dir):
    return {"S": fixed.s_measure_folder(str(directory), str(gt_dir))["mean_S"],
            "F": fixed.weighted_f_measure_folder(str(directory), str(gt_dir))["mean_Fw"],
            "E": fixed.e_measure_folder(str(directory), str(gt_dir))["mean_E"],
            "MAE": fixed.mae_folder(str(directory), str(gt_dir))["mean_MAE"]}


def infer(split, checkpoint, output, seal_dir):
    with (output.parent / f"{output.name}_infer.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "infer_seal.py"),
             "--seal-dir", str(seal_dir), "--split", split,
             "--checkpoint", str(checkpoint), "--output-dir", str(output),
             "--stage", "general"],
            check=True, stdout=log, stderr=subprocess.STDOUT,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT.parent / "Ours" / "evaluate.py", args.output_dir / "evaluate_source.py")
    rows = []
    for epoch in range(5, 36, 5):
        checkpoint = args.checkpoints / f"NESS_general_{epoch}.pth"
        prediction_dir = args.output_dir / f"val_epoch_{epoch}_maps"
        reused = args.reuse_dir / prediction_dir.name if args.reuse_dir else None
        if reused and complete(reused, args.seal_dir / "val" / "GT"):
            prediction_dir = reused
        else:
            infer("val", checkpoint, prediction_dir, args.seal_dir)
        if not complete(prediction_dir, args.seal_dir / "val" / "GT"):
            raise ValueError(f"Incomplete validation predictions: {prediction_dir}")
        metrics = {"F": fixed.weighted_f_measure_folder(str(prediction_dir), str(args.seal_dir / "val" / "GT"))["mean_Fw"],
                   "MAE": fixed.mae_folder(str(prediction_dir), str(args.seal_dir / "val" / "GT"))["mean_MAE"]}
        metrics = {key: float(value) for key, value in metrics.items()}
        rows.append({"epoch": epoch, "checkpoint": str(checkpoint), "predictions": str(prediction_dir), **metrics})
        (args.output_dir / "val_metrics.json").write_text(json.dumps(rows, indent=2))
        print(f"epoch {epoch}: {metrics}", flush=True)

    best = max(rows, key=lambda row: (row["F"], -row["MAE"]))
    checkpoint = args.checkpoints / f"NESS_general_{best['epoch']}.pth"
    (args.output_dir / "selection.json").write_text(json.dumps({"criterion": "val Fw descending, MAE ascending", "selected": best}, indent=2))
    prediction_dir = args.output_dir / "test_maps"
    infer("test", checkpoint, prediction_dir, args.seal_dir)
    if not complete(prediction_dir, args.seal_dir / "test" / "GT"):
        raise ValueError("Incomplete test predictions")
    metrics = {key: float(value) for key, value in evaluate(prediction_dir, args.seal_dir / "test" / "GT").items()}
    result = {"checkpoint": str(checkpoint), "val": best, "test": metrics}
    (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
