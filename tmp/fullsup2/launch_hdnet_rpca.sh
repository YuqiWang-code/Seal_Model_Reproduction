#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
mkdir -p $ROOT/logs
rm -rf $ROOT/HDNet-main/experiments/Seal_fullsup $ROOT/RPCANet-main/experiments/Seal_fullsup* $ROOT/HDNet-main/experiments/Seal_smoke $ROOT/RPCANet-main/experiments/Seal_smoke

cd $ROOT/HDNet-main
nohup $PY train_seal.py --device cuda:1 --epochs 300 --val-interval 20 --out experiments/Seal_fullsup > $ROOT/logs/hdnet_train.log 2>&1 &
echo "hdnet $!"

cd $ROOT/RPCANet-main
nohup $PY train_seal.py --device cuda:1 --net-name rpcanet --epochs 50 --val-interval 5 --out experiments/Seal_fullsup_rpcanet > $ROOT/logs/rpcanet_train.log 2>&1 &
echo "rpcanet $!"
nohup $PY train_seal.py --device cuda:1 --net-name rpcanet_pp --epochs 50 --val-interval 5 --out experiments/Seal_fullsup_rpcanetpp > $ROOT/logs/rpcanetpp_train.log 2>&1 &
echo "rpcanetpp $!"
sleep 90
for f in hdnet_train rpcanet_train rpcanetpp_train; do echo "== $f"; tail -2 $ROOT/logs/$f.log; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
