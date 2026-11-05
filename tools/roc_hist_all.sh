#!/bin/bash
# Build the 256-bin TP/FP histograms for the 12 fully-supervised baselines.
# Run on the server; each method writes one small JSON under results/roc_hist/.
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
GT=$ROOT/Seal_data/test/GT
OUT=$ROOT/results/roc_hist
mkdir -p "$OUT"

declare -A M=(
  [DNANet]='DNA-Net'
  [MSHNet]='MSHNet'
  [UIU-Net]='UIU-Net'
  [SCTransNet]='SCTransNet'
  [RPCANet]='RPCANet'
  [HDNet]='HDNet'
  [ILNet]='ILNet'
  [RPCANetpp]='RPCANet++'
  [L2SKNet]='L2SKNet'
  [BGM]='BGM'
  [DRPCANet]='DRPCA-Net'
  [DURD]='DURD'
)

for d in DNANet MSHNet UIU-Net SCTransNet RPCANet HDNet ILNet RPCANetpp L2SKNet BGM DRPCANet DURD; do
  maps="$ROOT/results/$d/Seal/fullsup_test_maps"
  if [ ! -d "$maps" ]; then
    echo "MISSING $maps"
    continue
  fi
  echo "== $d (${M[$d]})"
  "$PY" "$ROOT/tools/roc_hist.py" \
    --maps "$maps" --gt "$GT" \
    --out "$OUT/$d.json" --label "${M[$d]}"
done
echo ALL DONE
ls -la "$OUT"
