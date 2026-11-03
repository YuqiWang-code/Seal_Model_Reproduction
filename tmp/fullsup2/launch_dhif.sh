#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

# stop the RPCANet batch-SoftIoU probe (val IoU stayed 0)
PID=$(pgrep -f 'net-name rpcanet --epochs 15' | head -1)
[ -n "${PID:-}" ] && kill -9 $PID && echo "killed rpcanet_test1 $PID"
sleep 3

rm -rf $ROOT/DHiF-main/experiments/Seal_fullsup $ROOT/DHiF-main/experiments/Seal_smoke
cd $ROOT/DHiF-main
nohup $PY train_seal.py --device cuda:1 --epochs 100 --val-interval 10 --batch-size 12 \
  --out experiments/Seal_fullsup > $ROOT/logs/dhif_train.log 2>&1 &
echo "dhif $!"
sleep 90
tail -2 $ROOT/logs/dhif_train.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
