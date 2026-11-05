#!/bin/bash
# Target-level Pd/Fa curves for Ours + the 12 fully-supervised baselines.
# One process per method (48 cores available), each writes a small JSON.
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
GT=$ROOT/Seal_data/test/GT
OUT=$ROOT/results/target_curves
mkdir -p "$OUT" "$ROOT/logs"

run() {  # $1 = dir key, $2 = maps, $3 = label
  echo "start $3"
  "$PY" "$ROOT/tools/target_curves.py" --maps "$2" --gt "$GT" \
      --out "$OUT/$1.json" --label "$3" > "$ROOT/logs/tc_$1.log" 2>&1 \
      && echo "done $3" || echo "FAIL $3"
}

run Ours      "$ROOT/Ours/test_maps"                     Ours        &
run DNANet    "$ROOT/results/DNANet/Seal/fullsup_test_maps"    DNA-Net     &
run MSHNet    "$ROOT/results/MSHNet/Seal/fullsup_test_maps"    MSHNet      &
run UIU-Net   "$ROOT/results/UIU-Net/Seal/fullsup_test_maps"   UIU-Net     &
run SCTransNet "$ROOT/results/SCTransNet/Seal/fullsup_test_maps" SCTransNet &
run RPCANet   "$ROOT/results/RPCANet/Seal/fullsup_test_maps"   RPCANet     &
run HDNet     "$ROOT/results/HDNet/Seal/fullsup_test_maps"     HDNet       &
run ILNet     "$ROOT/results/ILNet/Seal/fullsup_test_maps"     ILNet       &
run RPCANetpp "$ROOT/results/RPCANetpp/Seal/fullsup_test_maps" RPCANet++   &
run L2SKNet   "$ROOT/results/L2SKNet/Seal/fullsup_test_maps"   L2SKNet     &
run BGM       "$ROOT/results/BGM/Seal/fullsup_test_maps"       BGM         &
run DRPCANet  "$ROOT/results/DRPCANet/Seal/fullsup_test_maps"  DRPCA-Net   &
run DURD      "$ROOT/results/DURD/Seal/fullsup_test_maps"      DURD        &
wait
echo ALL DONE
ls -la "$OUT"
