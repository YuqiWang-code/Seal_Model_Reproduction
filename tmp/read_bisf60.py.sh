#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction/results/Ours/BISF_60ep
for f in metrics4 metrics_small metrics4_bin metrics_small_bin; do
  echo "=== $f ==="
  /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python - "$R/$f.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if "metrics" in d:
    print({k: round(v, 8) for k, v in d["metrics"].items()})
    print({k: d["counts"][k] for k in ("tp", "fp", "fn", "n_gt", "n_pred", "n_detected", "false_alarm_pixels", "total_pixels")})
else:
    print({k: round(v, 10) for k, v in d["Seal"].items()})
PY
done
