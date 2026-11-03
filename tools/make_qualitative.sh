#!/bin/bash
# Generate binary maps for every method and rebuild the qualitative panels.
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
cd $ROOT

METHODS=(
  "DNANet=results/DNANet/Seal"
  "MSHNet=results/MSHNet/Seal"
  "UIU-Net=results/UIU-Net/Seal"
  "SCTransNet=results/SCTransNet/Seal"
  "RPCANet=results/RPCANet/Seal"
  "HDNet=results/HDNet/Seal"
  "ILNet=results/ILNet/Seal"
  "RPCANet++=results/RPCANetpp/Seal"
)
# optional extra methods passed as arguments: Name=results/Name/Seal
for extra in "$@"; do METHODS+=("$extra"); done

ARGS=()
for m in "${METHODS[@]}"; do
  name=${m%%=*}; dir=${m##*=}
  if [ ! -d "$ROOT/$dir/fullsup_test_maps" ]; then echo "skip $name (no predictions)"; continue; fi
  $PY tools/make_binary_maps.py --pred "$ROOT/$dir/fullsup_test_maps" --out "$ROOT/$dir/test_binary" | tail -1
  ARGS+=("$name=$ROOT/$dir/test_binary")
done

# Ours: the reference predictions of the paper
if [ -d "$ROOT/Ours/test_maps" ]; then
  $PY tools/make_binary_maps.py --pred "$ROOT/Ours/test_maps" --out "$ROOT/results/Ours/Seal/test_binary" | tail -1
  ARGS+=("Ours=$ROOT/results/Ours/Seal/test_binary")
fi

$PY tools/select_qualitative_samples.py --out $ROOT/docs/Seal定性对比样例.json | tail -3
$PY tools/build_qualitative_panels.py --samples $ROOT/docs/Seal定性对比样例.json \
  --methods "${ARGS[@]}" --out $ROOT/qualitative | tail -6
ls $ROOT/qualitative | head
