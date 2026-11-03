"""Write the cross-domain NESS-Net row into docs/Seal量化SOTA表.xlsx.

The main-table row must use the published refinement + NAMLab cross-domain
prediction (AGENTS.md, docs/NESS-Net_Seal实验记录.md), not the historical
Seal box-adaptation epoch5 result. Values come from
docs/小目标指标评估_20261103/NESS-Net.json so the table and the small-target
record cannot drift apart.

Usage:
    python tmp/apply_ness_cross_domain_row.py            # in place
    python tmp/apply_ness_cross_domain_row.py --target <copy.xlsx>
"""

import argparse
import json
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "docs" / "Seal量化SOTA表.xlsx"
SMALL_TARGET = ROOT / "docs" / "小目标指标评估_20261103" / "NESS-Net.json"

# S, Fw, E, MAE from docs/NESS-Net_Seal评估产物/published_metrics.json:refinement_namlab
ORIGINAL_FOUR = {
    "E": 0.4666185279426068,   # S  -> column E (S↑)
    "F": 0.001733883532895621,  # Fw -> column F
    "G": 0.263767112039326,    # E  -> column G (E↑)
    "H": 0.042296525090932846,  # MAE -> column H
}
SMALL_FOUR = {"I": "IoU", "J": "F1", "K": "Pd", "L": "Fa"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(SMALL_TARGET.read_text(encoding="utf-8"))["metrics"]
    wb = openpyxl.load_workbook(args.target)
    ws = wb.worksheets[0]
    row = 17

    values = {"A": "NESS-Net", "D": "Scribble", "O": "公开refinement+NAMLab跨域测试；T替代深度"}
    values.update(ORIGINAL_FOUR)
    values.update({col: metrics[name] for col, name in SMALL_FOUR.items()})

    current = {col: ws[f"{col}{row}"].value for col in values}
    if current == values:
        print(f"{args.target} row {row} already up to date")
        return
    if args.check_only:
        raise SystemExit(f"row {row} differs from the cross-domain values: {current}")
    for col, value in values.items():
        ws[f"{col}{row}"] = value
    wb.save(args.target)
    print(f"updated {args.target} row {row}:")
    print([ws.cell(row, c).value for c in range(1, 16)])


if __name__ == "__main__":
    main()
