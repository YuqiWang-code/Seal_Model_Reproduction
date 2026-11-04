#!/bin/bash
# Measure params / GFLOPs / FPS for the 12 fully-supervised baselines.
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
OUT=$ROOT/results/complexity
mkdir -p $OUT
run () {
  dir=$1; kind=$2
  cd $ROOT/$dir
  $PY $ROOT/tools/measure_complexity_one.py --kind $kind --device cuda:1 > $OUT/$kind.json 2> $OUT/$kind.err
  rc=$?
  printf '%-12s rc=%s %s\n' "$kind" "$rc" "$(tail -1 $OUT/$kind.json 2>/dev/null | cut -c1-140)"
  [ $rc -ne 0 ] && tail -3 $OUT/$kind.err
}
run DNANet-main dnanet
run MSHNet-main mshnet
run UIU-Net-main uiunet
run SCTransNet-main sctransnet
run RPCANet-main rpcanet
run HDNet-main hdnet
run ILNet-main ilnet
run RPCANet-main rpcanetpp
run L2SKNet-main l2sknet
run BGM-main bgm
run DRPCA-Net-main drpcanet
run DURD-main durd
echo "== results =="
cat $OUT/*.json
