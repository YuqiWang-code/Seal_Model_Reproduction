"""把 Ours 的两行结果写进 docs/Seal量化SOTA表.xlsx（第 33 行留空，34/35 两行）。

数值来源：每个方法目录下的 ``metrics4.json``（根目录 evaluate.py 的四个原函数）与
``metrics_small.json``（evaluate_small_targets.py 的 IoU/F1/Pd/Fa）。
列按表头标题定位，不依赖列号；样式从第 32 行（最后一个全监督基线）复制。

用法：
    python tmp/fill_ours_rows.py                       # 写 docs/Seal量化SOTA表.xlsx
    python tmp/fill_ours_rows.py --target <copy.xlsx>  # 写副本（可用于演练）
    python tmp/fill_ours_rows.py --original-m4 ... --original-small ... \
        --fullsup-m4 ... --fullsup-small ...           # 指定 JSON 路径
"""

import argparse
import json
from copy import copy
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "Seal量化SOTA表.xlsx"
DEFAULT = {
    "original_m4": "results/Ours/Original/metrics4.json",
    "original_small": "results/Ours/Original/metrics_small.json",
    "fullsup_m4": "results/Ours/FullySup/metrics4.json",
    "fullsup_small": "results/Ours/FullySup/metrics_small.json",
}
HEADERS = {"S↑": "S", "Fw↑": "Fw", "E↑": "E", "MAE↓": "MAE",
           "IoU": "IoU", "F1": "F1", "Pd": "Pd", "Fa": "Fa"}
ROWS = [
    ("Ours（原始版）", "Box", "Seal train 框监督（GT 框作提示）→ Seal test"),
    ("Ours（全监督版）", "Full GT", "Seal train 全像素 GT → Seal test（框仅作 Decoder 提示）"),
]


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def four_metrics(payload: dict) -> dict:
    """evaluate.py 输出 {"Seal": {"S","F","E","MAE"}}；兼容直接平铺。"""
    body = payload.get("Seal", payload)
    return {"S": float(body["S"]), "Fw": float(body["F"]), "E": float(body["E"]), "MAE": float(body["MAE"])}


def small_metrics(payload: dict) -> dict:
    m = payload["metrics"]
    return {k: float(m[k]) for k in ("IoU", "F1", "Pd", "Fa")}


def locate_columns(ws) -> dict:
    columns = {}
    for r in (1, 2):
        for c in range(1, ws.max_column + 1):
            value = ws.cell(r, c).value
            if isinstance(value, str) and value.strip() in HEADERS:
                columns[HEADERS[value.strip()]] = c
    missing = [k for k in ("S", "Fw", "E", "MAE", "IoU", "F1", "Pd", "Fa") if k not in columns]
    if missing:
        raise RuntimeError(f"表头缺少列: {missing}")
    return columns


def copy_style(src, dst) -> None:
    dst._style = copy(src._style)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=TABLE)
    parser.add_argument("--original-m4", default=None)
    parser.add_argument("--original-small", default=None)
    parser.add_argument("--fullsup-m4", default=None)
    parser.add_argument("--fullsup-small", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = {
        "original_m4": Path(args.original_m4 or ROOT / DEFAULT["original_m4"]),
        "original_small": Path(args.original_small or ROOT / DEFAULT["original_small"]),
        "fullsup_m4": Path(args.fullsup_m4 or ROOT / DEFAULT["fullsup_m4"]),
        "fullsup_small": Path(args.fullsup_small or ROOT / DEFAULT["fullsup_small"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"缺少 {name}: {path}")

    rows = [
        {**four_metrics(load_json(paths["original_m4"])), **small_metrics(load_json(paths["original_small"]))},
        {**four_metrics(load_json(paths["fullsup_m4"])), **small_metrics(load_json(paths["fullsup_small"]))},
    ]
    for (name, _, _), values in zip(ROWS, rows):
        print(f"{name}: " + " ".join(f"{k}={v:.6g}" for k, v in values.items()))

    wb = openpyxl.load_workbook(args.target)
    ws = wb.worksheets[0]
    columns = locate_columns(ws)

    if ws.max_row < 32:
        raise SystemExit(f"表只有 {ws.max_row} 行，预期至少 32 行（12 个全监督基线）")
    last_row = 32
    for r in range(3, 33):
        if ws.cell(r, 1).value is None:
            raise SystemExit(f"第 {r} 行方法名为空，表结构异常")

    # 第 33 行留空：清掉可能残留的内容，样式沿用第 32 行
    blank_row = 33
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(blank_row, c)
        cell.value = None
        copy_style(ws.cell(last_row, c), cell)

    for offset, ((name, supervision, note), values) in enumerate(zip(ROWS, rows), start=1):
        row = blank_row + offset
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row, c)
            cell.value = None
            copy_style(ws.cell(last_row, c), cell)
        ws.cell(row, 1).value = name
        ws.cell(row, 2).value = "Ours"
        ws.cell(row, 3).value = "RGB-T"
        ws.cell(row, 4).value = supervision
        for key, value in values.items():
            ws.cell(row, columns[key]).value = value
        ws.cell(row, 15).value = note

    if args.dry_run:
        print("dry-run：未写入")
        return
    wb.save(args.target)
    print(f"已写入 {args.target} 第 33（空）、34、35 行")

    check = openpyxl.load_workbook(args.target, data_only=True).worksheets[0]
    print("rows", check.max_row)
    for r in (33, 34, 35):
        print(r, [check.cell(r, c).value for c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)])


if __name__ == "__main__":
    main()
