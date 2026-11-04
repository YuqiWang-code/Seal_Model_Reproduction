"""把八项指标写进 docs/Seal量化SOTA表.xlsx 的 Ablation 表某一方法行。

默认写第 15 行 `B+BISF`（对照第 14 行 `B`）。列按表头标题定位：
B..J = S/Fw/E/MAE/IoU/F1/Pd/Fa(原始比例)/Fa×10⁻⁶，样式与数值精度沿用同区的 `B` 行。

用法：
    python tmp/fill_ablation_row.py --m4 <metrics4.json> --small <metrics_small.json>
    python tmp/fill_ablation_row.py --target <copy.xlsx> --row 15 ...
    python tmp/fill_ablation_row.py --dry-run ...
"""

import argparse
import json
from copy import copy
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "Seal量化SOTA表.xlsx"
FIELDS = ["S", "Fw", "E", "MAE", "IoU", "F1", "Pd", "Fa"]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def four(payload: dict) -> dict:
    body = payload.get("Seal", payload)
    return {"S": float(body["S"]), "Fw": float(body["F"]), "E": float(body["E"]), "MAE": float(body["MAE"])}


def small(payload: dict) -> dict:
    m = payload["metrics"]
    return {k: float(m[k]) for k in ("IoU", "F1", "Pd", "Fa")}


def header_columns(ws, header_row: int) -> dict:
    mapping = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(header_row, c).value
        if not isinstance(v, str):
            continue
        key = v.strip()
        if key in ("S↑", "Fw↑", "E↑", "MAE↓", "IoU↑", "F1↑", "Pd↑"):
            mapping[{"S↑": "S", "Fw↑": "Fw", "E↑": "E", "MAE↓": "MAE",
                     "IoU↑": "IoU", "F1↑": "F1", "Pd↑": "Pd"}[key]] = c
        elif key.startswith("Fa↓") and "10" in key:
            mapping["Fa_x1e6"] = c
        elif key.startswith("Fa↓"):
            mapping["Fa"] = c
        elif key == "Seal":
            mapping["seal"] = c
    missing = [k for k in FIELDS + ["Fa_x1e6"] if k not in mapping]
    if missing:
        raise RuntimeError(f"表头缺少列: {missing}")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=TABLE)
    ap.add_argument("--sheet", default="Ablation")
    ap.add_argument("--row", type=int, default=15, help="要写入的行（默认 B+BISF）")
    ap.add_argument("--baseline-row", type=int, default=14, help="样式与精度参照的行（默认 B）")
    ap.add_argument("--m4", type=Path, required=True)
    ap.add_argument("--small", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    values = {**four(load_json(args.m4)), **small(load_json(args.small))}
    print("values:", " ".join(f"{k}={v:.6g}" for k, v in values.items()))

    wb = openpyxl.load_workbook(args.target)
    ws = wb[args.sheet]
    cols = header_columns(ws, args.baseline_row - 1)  # 表头在参照行的上一行

    # 精度沿用同区 B 行：S/Fw/E/MAE/IoU/F1/Pd 五位小数，Fa 原始比例 8 位，Fa×10⁻⁶ 一位
    rendered = {k: round(values[k], 5) for k in ("S", "Fw", "E", "MAE", "IoU", "F1", "Pd")}
    rendered["Fa"] = round(values["Fa"], 8)
    rendered["Fa_x1e6"] = round(values["Fa"] * 1e6, 1)
    print("rendered:", rendered)

    for c in range(1, ws.max_column + 1):
        dst = ws.cell(args.row, c)
        src = ws.cell(args.baseline_row, c)
        dst._style = copy(src._style)

    for key, col in cols.items():
        if key == "seal":
            continue
        ws.cell(args.row, col).value = rendered[key]

    if args.dry_run:
        print("dry-run：未写入")
    else:
        wb.save(args.target)
        print(f"已写入 {args.target} [{args.sheet}] 第 {args.row} 行")

    check = openpyxl.load_workbook(args.target, data_only=True)[args.sheet]
    for r in (args.baseline_row, args.row):
        print(r, [check.cell(r, c).value for c in range(1, check.max_column + 1)])


if __name__ == "__main__":
    main()
