# 作者：恽劼恒
# 时间：2026/11/04
import argparse
import json
import os
import sys
import cv2
import evaluate
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from glob import glob
from pathlib import Path
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics import AS1000Metrics


METRIC_ORDER = ("S", "F", "E", "MAE", "IoU", "F1", "Pd", "Fa")
DEFAULT_GT_DIR = "/share_datasets/Seal_Model_Reproduction/Seal_data/test/GT"


def round_half_up(value, places=5):
    """四舍五入到指定小数位，返回固定宽度的字符串。"""
    quant = Decimal("1").scaleb(-places)
    number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{number:.{places}f}"


def detection_metrics(pred_dir, gt_dir, match_distance=3):
    """用 metrics.AS1000Metrics 计算 IoU、F1、Pd、Fa。预测图为 0~255 的概率。"""
    pred_files = sorted(glob(os.path.join(pred_dir, "*")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*")))
    if not gt_files:
        raise RuntimeError(f"No ground-truth files found: {gt_dir}")

    gt_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in gt_files}
    metric = AS1000Metrics(match_distance=match_distance)
    matched = 0

    for pred_path in tqdm(pred_files, desc="Computing IoU/F1/Pd/Fa"):
        name = os.path.splitext(os.path.basename(pred_path))[0].lower()
        gt_path = gt_map.get(name)
        if gt_path is None:
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            raise RuntimeError(f"Failed to load: {os.path.basename(pred_path)}")

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        probability = np.clip(pred.astype(np.float32) / 255.0, 0.0, 1.0)
        mask = gt > 128
        metric.update(probability, mask)
        matched += 1

    if matched == 0:
        raise RuntimeError(f"No matched prediction/GT pairs in {pred_dir} and {gt_dir}")

    return metric.get()


def evaluate_all(pred_root, match_distance=3, gt_dir=DEFAULT_GT_DIR):
    saliency = evaluate.evaluate_all(pred_root, gt_dir=gt_dir)
    detection = {"Seal": detection_metrics(pred_root, gt_dir, match_distance=match_distance)}

    dataset_names = list(dict.fromkeys([*saliency, *detection]))
    results = {}
    for name in dataset_names:
        merged = {}
        merged.update(saliency.get(name, {}))
        merged.update(detection.get(name, {}))
        missing = [metric for metric in METRIC_ORDER if metric not in merged]
        if missing:
            raise RuntimeError(f"{name} 缺少指标: {', '.join(missing)}")
        results[name] = {metric: merged[metric] for metric in METRIC_ORDER}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate S / F / E / MAE / IoU / F1 / Pd / Fa")
    parser.add_argument("--pred_root", type=str, default="/share_datasets/Seal_Model_Reproduction/results/Ours/BISF/test_maps")
    parser.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR)
    parser.add_argument("--match_distance", type=float, default=3)
    parser.add_argument("--json_output", type=str, default=None, help="可选：把完整精度结果写到 JSON")
    args = parser.parse_args()

    results = evaluate_all(args.pred_root, match_distance=args.match_distance, gt_dir=args.gt_dir)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    formatted_results = {
        dataset: {
            metric: round_half_up(value, 7 if metric == "Fa" else 5)
            for metric, value in metrics.items()
        }
        for dataset, metrics in results.items()
    }
    print(json.dumps(formatted_results, ensure_ascii=False, indent=2))
