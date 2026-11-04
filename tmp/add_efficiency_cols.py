"""Insert Params↓ / GFLOPs↓ / FPS↑ right after the Fa column (L).

Only three blank columns are added; every existing value, style, hyperlink and
merged header keeps its meaning and simply moves three columns right from 论文
onwards. No new group header is invented and no metric is filled in.

Why not ws.insert_cols: openpyxl moves values/styles but not the cell
hyperlinks (the 论文/代码 cells link to the paper/repo), which corrupts them.
Here cells are shifted by hand so hyperlinks move with their cells.
Rerunning is a no-op.
"""

import argparse
import copy
from pathlib import Path

import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs" / "Seal量化SOTA表.xlsx"
NEW_HEADERS = ["Params↓", "GFLOPs↓", "FPS↑"]
FORMATS = {"Params↓": "0.00", "GFLOPs↓": "0.00", "FPS↑": "0.0"}
WIDTHS = (11.0, 12.0, 9.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.path)
    ws = wb.worksheets[0]
    row2 = [ws.cell(2, c).value for c in range(1, ws.max_column + 1)]
    if "Params↓" in row2:
        print("already present, nothing to do")
        return

    fa = row2.index("Fa") + 1
    at = fa + 1
    maxr, maxc = ws.max_row, ws.max_column
    merges = [(m.min_row, m.min_col, m.max_row, m.max_col) for m in ws.merged_cells.ranges]
    for m in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(m))

    # shift everything from the 论文 column rightwards, hyperlinks included
    for c in range(maxc, at - 1, -1):
        for r in range(1, maxr + 1):
            src, dst = ws.cell(r, c), ws.cell(r, c + 3)
            dst.value = src.value
            dst._style = copy.copy(src._style)
            if src.hyperlink is not None:
                dst.hyperlink = Hyperlink(ref=dst.coordinate,
                                          target=src.hyperlink.target,
                                          location=src.hyperlink.location,
                                          tooltip=src.hyperlink.tooltip,
                                          display=src.hyperlink.display)
                src.hyperlink = None
            src.value = None

    for offset, name in enumerate(NEW_HEADERS):
        cell = ws.cell(2, at + offset)
        cell.value = name
        cell._style = copy.copy(ws.cell(2, fa)._style)
        for r in range(3, maxr + 1):
            body = ws.cell(r, at + offset)
            body.value = None
            body.hyperlink = None
            body._style = copy.copy(ws.cell(r, fa)._style)
            body.number_format = FORMATS[name]

    for r1, c1, r2, c2 in merges:
        ws.merge_cells(start_row=r1, start_column=c1 if c1 < at else c1 + 3,
                       end_row=r2, end_column=c2 if c2 < at else c2 + 3)

    widths = {column_index_from_string(k): v.width for k, v in ws.column_dimensions.items() if v.width}
    for idx in sorted(widths, reverse=True):
        if idx >= at:
            old, new = get_column_letter(idx), get_column_letter(idx + 3)
            ws.column_dimensions[new].width = widths[idx]
            if old in ws.column_dimensions:
                del ws.column_dimensions[old]
    for offset, width in enumerate(WIDTHS):
        ws.column_dimensions[get_column_letter(at + offset)].width = width

    wb.save(args.out or args.path)
    print("saved", args.out or args.path)
    print("row1:", [(c.column_letter, c.value) for c in ws[1] if c.value])
    print("row2:", [(c.column_letter, c.value) for c in ws[2] if c.value])
    print("merged:", sorted(str(m) for m in ws.merged_cells.ranges))


if __name__ == "__main__":
    main()
