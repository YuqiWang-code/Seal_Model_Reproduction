#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
cd $ROOT/SCTransNet-main
$PY train_seal.py --device cuda:1 --epochs 1 --val-interval 1 --batch-size 4 --workers 4 \
  --out experiments/Seal_smoke > /tmp/sc_smoke.log 2>&1
echo "rc=$?"
tail -12 /tmp/sc_smoke.log
