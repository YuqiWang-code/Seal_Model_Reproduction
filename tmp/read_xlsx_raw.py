"""Minimal xlsx reader (values only) for files openpyxl cannot parse.

The 腾讯文档 export has a stylesheet openpyxl rejects ("Fill() takes no
arguments"), so values are read straight from the sheet XML instead. Nothing
is written.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
COORD = re.compile(r"([A-Z]+)(\d+)")


def col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def load_shared(root: Path):
    path = root / "xl" / "sharedStrings.xml"
    if not path.exists():
        return []
    tree = ET.parse(path)
    out = []
    for si in tree.getroot().findall("m:si", NS):
        out.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    return out


def read_sheet(path: Path, shared):
    tree = ET.parse(path)
    root = tree.getroot()
    values = {}
    for cell in root.iter(f"{{{NS['m']}}}c"):
        ref = cell.get("r")
        m = COORD.match(ref)
        if not m:
            continue
        kind = cell.get("t")
        v = cell.find("m:v", NS)
        is_ = cell.find("m:is", NS)
        if kind == "s" and v is not None:
            value = shared[int(v.text)]
        elif kind == "inlineStr" and is_ is not None:
            value = "".join(t.text or "" for t in is_.iter(f"{{{NS['m']}}}t"))
        elif kind in ("str", "e") and v is not None:
            value = v.text
        elif v is not None:
            text = v.text or ""
            try:
                value = float(text)
                if value.is_integer() and "." not in text and "E" not in text.upper():
                    value = int(value)
            except ValueError:
                value = text
        else:
            value = None
        if value not in (None, ""):
            values[(int(m.group(2)), col_index(m.group(1)))] = value
    merges = []
    for mc in root.iter(f"{{{NS['m']}}}mergeCell"):
        merges.append(mc.get("ref"))
    return values, merges


def dump(path: Path, shared, name=""):
    values, merges = read_sheet(path, shared)
    rows = sorted({r for r, _ in values})
    cols = sorted({c for _, c in values})
    print(f"--- {name or path.name}: {len(values)} non-empty cells, rows {rows[0]}..{rows[-1]}, cols 1..{cols[-1]}")
    print("merged:", merges)
    for r in rows:
        cells = [(c, values[(r, c)]) for c in cols if (r, c) in values]
        print(r, cells)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    shared = load_shared(root)
    for sheet in sys.argv[2:]:
        dump(root / "xl" / "worksheets" / sheet, shared, sheet)
