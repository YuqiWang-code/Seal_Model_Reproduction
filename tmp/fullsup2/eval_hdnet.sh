#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
L=$ROOT/logs
echo "== hdnet val"
grep val_iou $L/hdnet_train.log | tail -3
PID=$(pgrep -f 'HDNet-main' | head -1 || true)
if [ -n "${PID:-}" ]; then kill -9 $PID && echo "killed hdnet $PID"; fi
sleep 3
ls $ROOT/HDNet-main/experiments/Seal_fullsup/
cd $ROOT
DIR=HDNet-main NAME=HDNet GPU=1 bash /tmp/run_eval2.sh 2>&1 | tail -12
