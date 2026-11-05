"""把 Ours/CODE 实测的 Params / GFLOPs / FPS 填进 docs/Seal量化SOTA表.xlsx 的 Ours 行。

按列标题定位（Params↓ / GFLOPs↓ / FPS↑），按方法名定位行，样式与数字格式取自相邻的
DNA-Net 行（第 21 行），不依赖任何硬编码列号。
"""

from __future__ import annotations

import argparse
import shutil
from copy import copy
from pathlib import Path

import openpyxl

XLSX = Path("docs/Seal量化SOTA表.xlsx")
HEADERS = ("Params↓", "GFLOPs↓", "FPS↑")


def find_col(ws, title: str) -> int:
    for col in range(1, ws.max_column + 1):
        value = ws.cell(2, col).value
        if isinstance(value, str) and value.strip() == title:
            return col
    raise KeyError(f"表头里找不到 {title}")


def find_row(ws, method: str) -> int:
    for row in range(3, ws.max_row + 1):
        value = ws.cell(row, 1).value
        if isinstance(value, str) and value.strip() == method:
            return row
    raise KeyError(f"找不到方法行 {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True, help="全部 nn.Parameter，单位 M")
    ap.add_argument("--gflops", type=float, required=True, help="乘加次数 / 1e9")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--method", default="Ours")
    ap.add_argument("--style-from", default="DNA-Net", help="样式参照的方法行")
    ap.add_argument("--note-append", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wb = openpyxl.load_workbook(XLSX)
    ws = wb["Quantity"]
    row = find_row(ws, args.method)
    style_row = find_row(ws, args.style_from)
    cols = {title: find_col(ws, title) for title in HEADERS}

    values = {"Params↓": args.params, "GFLOPs↓": args.gflops, "FPS↑": args.fps}
    for title, col in cols.items():
        old = ws.cell(row, col).value
        src = ws.cell(style_row, col)
        cell = ws.cell(row, col)
        cell.value = round(values[title], 3 if title != "FPS↑" else 2)
        cell.number_format = src.number_format
        cell.font = copy(src.font)
        cell.alignment = copy(src.alignment)
        cell.border = copy(src.border)
        cell.fill = copy(src.fill)
        print(f"{title:10s} col {col:2d}  {old!r} -> {cell.value!r} fmt={cell.number_format!r}")

    if args.note_append:
        note_col = find_col(ws, "重训/测试")
        cell = ws.cell(row, note_col)
        old = cell.value or ""
        cell.value = old.rstrip("；; ") + "；" + args.note_append if old else args.note_append
        print(f"备注 col {note_col}: {cell.value}")

    if args.dry_run:
        print("dry-run，不写盘")
        return
    tmp = XLSX.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    try:
        XLSX.unlink()
        tmp.rename(XLSX)
    except PermissionError:
        shutil.copyfile(tmp, XLSX.with_name("Seal量化SOTA表.填表备份.xlsx"))
        print("原文件被占用，已写入副本 docs/Seal量化SOTA表.填表备份.xlsx")
        raise SystemExit(1)
    print(f"已写入 {XLSX}")


if __name__ == "__main__":
    main()
