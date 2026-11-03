"""Fix two cell defects in docs/Seal量化SOTA表.xlsx (idempotent).

  * A13: 'FCT-SAM（直接测)\\n'  ->  'FCT-SAM（直接测）'   (stray newline, mismatched bracket)
  * D17: 'Box'                  ->  'Scribble'          (row 17 now uses the published
                                                        NESS-Net refinement weights, whose
                                                        source supervision is scribble;
                                                        'Box' was left from the removed
                                                        Seal box-adaptation row)

Every other cell is left untouched; the script prints a diff of the two cells only.
"""

from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'docs' / 'Seal量化SOTA表.xlsx'

FIXES = {
    'A13': 'FCT-SAM（直接测）',
    'D17': 'Scribble',
}


def main():
    wb = openpyxl.load_workbook(PATH)
    ws = wb.worksheets[0]
    changed = False
    for ref, new in FIXES.items():
        old = ws[ref].value
        if old == new:
            print(f'{ref}: already {new!r}')
            continue
        print(f'{ref}: {old!r} -> {new!r}')
        ws[ref] = new
        changed = True
    if changed:
        wb.save(PATH)
        print('saved')
    else:
        print('nothing to change')


if __name__ == '__main__':
    main()
