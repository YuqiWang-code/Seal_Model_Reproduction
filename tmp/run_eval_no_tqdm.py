"""Run the four original functions without installing tqdm.

The repository's evaluate.py imports tqdm only for the progress bar; a local
stub keeps the metric functions untouched (they are copied verbatim from
Ours/evaluate.py and must not be edited).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp' / 'stub'))
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402  (project root entry)

pred_root, gt_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
result = evaluate.evaluate_all(pred_root, gt_dir)
Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
