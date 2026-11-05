#!/bin/bash
# Object-level PR for the 14 Seal methods (parallel, ~10 min).
#   bash tools/target_pr_all.sh            # writes results/target_pr/<stem>.json
# then plot with tools/plot_pr_object.py --manifest tmp/roc/manifest_target_pr.json
set -u
cd /share_datasets/Seal_Model_Reproduction || exit 1
PY=${PY:-/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python}
GT=Seal_data/test/GT
OUT=results/target_pr
LOG=$OUT/logs
mkdir -p "$OUT" "$LOG"

run() {
  local label="$1" maps="$2" stem="$3"
  "$PY" tools/target_pr.py --maps "$maps" --gt "$GT" \
      --out "$OUT/$stem.json" --label "$label" \
      > "$LOG/$stem.log" 2>&1
  echo "$stem done rc=$?"
}

run "DNA-Net"    results/DNANet/Seal/fullsup_test_maps     DNANet     &
run "UIU-Net"    results/UIU-Net/Seal/fullsup_test_maps    UIU-Net    &
run "MSHNet"     results/MSHNet/Seal/fullsup_test_maps     MSHNet     &
run "SCTransNet" results/SCTransNet/Seal/fullsup_test_maps SCTransNet &
run "RPCANet"    results/RPCANet/Seal/fullsup_test_maps    RPCANet    &
run "HDNet"      results/HDNet/Seal/fullsup_test_maps      HDNet      &
run "ILNet"      results/ILNet/Seal/fullsup_test_maps      ILNet      &
run "L2SKNet"    results/L2SKNet/Seal/fullsup_test_maps    L2SKNet    &
run "BGM"        results/BGM/Seal/fullsup_test_maps        BGM        &
run "DRPCA-Net"  results/DRPCANet/Seal/fullsup_test_maps   DRPCANet   &
run "RPCANet++"  results/RPCANetpp/Seal/fullsup_test_maps  RPCANetpp  &
run "DURD"       results/DURD/Seal/fullsup_test_maps       DURD       &
wait
echo "baselines done"
run "SAM3" Ours/SAM3-test_maps SAM3 &
run "Ours" Ours/test_maps      Ours1105 &
wait
echo "all done"
ls -1 "$OUT"/*.json | wc -l
