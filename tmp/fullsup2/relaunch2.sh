#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
rm -rf $ROOT/HDNet-main/experiments/Seal_fullsup $ROOT/DHiF-main/experiments/Seal_fullsup \
       $ROOT/RPCANet-main/experiments/Seal_fullsup_rpcanetpp2

cd $ROOT/HDNet-main
nohup $PY train_seal.py --device cuda:1 --epochs 120 --val-interval 10 --out experiments/Seal_fullsup > $ROOT/logs/hdnet_train.log 2>&1 &
echo "hdnet $!"
cd $ROOT/DHiF-main
nohup $PY train_seal.py --device cuda:1 --epochs 100 --val-interval 10 --batch-size 8 --out experiments/Seal_fullsup > $ROOT/logs/dhif_train.log 2>&1 &
echo "dhif $!"
cd $ROOT/RPCANet-main
nohup $PY train_seal.py --device cuda:1 --net-name rpcanet_pp --epochs 50 --val-interval 5 --out experiments/Seal_fullsup_rpcanetpp2 > $ROOT/logs/rpcanetpp2_train.log 2>&1 &
echo "rpcanetpp2 $!"
sleep 150
for f in hdnet_train dhif_train rpcanetpp2_train; do echo "== $f"; tail -2 $ROOT/logs/$f.log; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
