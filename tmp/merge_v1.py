"""Merge the new content of docs/Seal量化SOTA表_v1.xlsx into the project table.

What is merged (nothing existing is rewritten):
  * headers IoU/F1/Pd get the ↑ used by the source table;
  * three new rows appended after the 12 full-supervision baselines:
      SAM3            (ICLR26, RGB, full GT, values from the source table)
      B+Fusion        (RGB-T, full GT, values recomputed at full precision)
      多模态LoRA+Fusion (name only, results pending)
  * a new "Ablation" sheet with the CM-LoRA / BISF / Total blocks.
    Values stay in the project convention: Fa is the raw ratio; the ×1e-6
    column is added separately and labelled, because the source header
    "Fa↓×10-6（8位）" sits on raw values.
"""

import argparse
import copy
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "Seal量化SOTA表.xlsx"

NEW_ROWS = [
    {
        "A": "SAM3", "B": "ICLR26", "C": "RGB", "D": "Full GT",
        "E": 0.49795, "F": 0.51296, "G": 0.64987, "H": 0.00055,
        "I": 0.22472, "J": 0.36698, "K": 0.68793, "L": 0.0003508,
        "P": "SAM 3: Segment Anything with Concepts",
        "Q": "https://github.com/facebookresearch/sam3",
        "R": "官方权重测试",
    },
    {
        "A": "B+Fusion", "B": None, "C": "RGB-T", "D": "Full GT",
        "E": 0.4991906041914764, "F": 0.5731357887666101,
        "G": 0.6932751447227452, "H": 0.0002285943628521636,
        "I": 0.4605456095481671, "J": 0.6306487199542382,
        "K": 0.8733577136665313, "L": 2.9846095055794e-05,
        "P": "本文方法（B + 融合）", "Q": None,
        "R": "Seal train/val 全监督重训 → Seal test",
    },
    {
        "A": "多模态LoRA+Fusion", "B": None, "C": "RGB-T", "D": "Full GT",
        "P": "本文方法（多模态 LoRA + 融合），训练中", "Q": None, "R": None,
    },
]

# Ablation blocks copied from the source sheet (raw Fa in column I, x1e-6 in J)
ABLATION = [
    ("CM-LoRA", [
        ("RGB", 0.49795, 0.51296, 0.64987, 0.00055, 0.22472, 0.36698, 0.68793, 0.0003508),
        ("RGB+LoRA", 0.49925, 0.70397, 0.68471, 0.00015, 0.4772, 0.64609, 0.90912, 1.34e-05),
        ("T", 0.49833, 0.59378, 0.67292, 0.00062, 0.22456, 0.36676, 0.78261, 0.0003221),
        ("T+LoRA", 0.49907, 0.61202, 0.67175, 0.00021, 0.37384, 0.54423, 0.85182, 2.67e-05),
        ("RGB-T", 0.49897, 0.63763, 0.67956, 0.00023, 0.4216, 0.59314, 0.8143, 5.57e-05),
        ("3LoRA",),
    ]),
    ("BISF", [
        ("B", 0.49897, 0.63763, 0.67956, 0.00023, 0.4216, 0.59314, 0.8143, 5.57e-05),
        ("B+BISF",),
    ]),
    ("Total", [
        ("B", 0.49897, 0.63763, 0.67956, 0.00023, 0.4216, 0.59314, 0.8143, 5.57e-05),
        ("B+CM-LoRA",),
        ("B+BISF",),
    ]),
]

METRIC_HEADERS = ["S↑", "Fw↑", "E↑", "MAE↓", "IoU↑", "F1↑", "Pd↑", "Fa↓（原始比例）", "Fa↓×10⁻⁶"]
THIN = Side(style="thin", color="FF999999")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD_FILL = PatternFill("solid", fgColor="FFF9E3")
TITLE_FILL = PatternFill("solid", fgColor="FFDCE6F1")


def style_from(cell, source):
    cell._style = copy.copy(source._style)


def build_ablation(wb, ws_main):
    ws = wb.create_sheet("Ablation")
    row = 1
    for title, entries in ABLATION:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
        cell = ws.cell(row, 1, title)
        cell.font = Font(bold=True, size=12)
        cell.fill = TITLE_FILL
        cell.alignment = Alignment(horizontal="left")
        row += 1
        ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=1)
        ws.cell(row, 1, "Modules").font = Font(bold=True)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=10)
        seal = ws.cell(row, 2, "Seal")
        seal.font = Font(bold=True)
        seal.alignment = Alignment(horizontal="center")
        row += 1
        for col, name in enumerate(METRIC_HEADERS, start=2):
            head = ws.cell(row, col, name)
            head.font = Font(bold=True)
            head.fill = HEAD_FILL
            head.alignment = Alignment(horizontal="center")
            head.border = BORDER
        row += 1
        for entry in entries:
            name = entry[0]
            ws.cell(row, 1, name).border = BORDER
            if len(entry) > 1:
                values = entry[1:]
                for col, value in enumerate(values, start=2):
                    cell = ws.cell(row, col, value)
                    cell.border = BORDER
                    cell.number_format = "0.00000" if col < 10 else "0.00000000"
                ws.cell(row, 10, round(values[-1] * 1e6, 4)).number_format = "0.0"
            row += 1
        row += 1
    ws.cell(row, 1, "Ours")
    for col in range(1, 11):
        ws.column_dimensions[get_column_letter(col)].width = 16 if col > 1 else 22
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=TABLE)
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.table)
    ws = wb.worksheets[0]
    header = {str(ws.cell(2, c).value).strip(): c for c in range(1, ws.max_column + 1) if ws.cell(2, c).value}
    for old, new in (("IoU", "IoU↑"), ("F1", "F1↑"), ("Pd", "Pd↑")):
        if old in header:
            ws.cell(2, header[old]).value = new

    # A-D are merged column headers (no value in row 2), so they use fixed indices
    target_col = {"A": 1, "B": 2, "C": 3, "D": 4}
    for letter, name in (("E", "S↑"), ("F", "Fw↑"), ("G", "E↑"), ("H", "MAE↓"),
                         ("I", "IoU"), ("J", "F1"), ("K", "Pd"), ("L", "Fa"),
                         ("P", "论文"), ("Q", "代码"), ("R", "重训/测试")):
        target_col[letter] = header[name]
    last = ws.max_row
    for offset, row in enumerate(NEW_ROWS, start=1):
        r = last + offset
        for target in range(1, ws.max_column + 1):
            ws.cell(r, target)._style = copy.copy(ws.cell(last, target)._style)
        for letter, value in row.items():
            if value is None:
                continue
            ws.cell(r, target_col[letter], value)

    if "Ablation" not in wb.sheetnames:
        build_ablation(wb, ws)
    wb.save(args.table)
    print(f"updated {args.table}: rows {last + 1}..{last + len(NEW_ROWS)}, sheets {wb.sheetnames}")


if __name__ == "__main__":
    main()
