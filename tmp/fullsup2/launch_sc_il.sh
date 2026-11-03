#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
rm -rf $ROOT/SCTransNet-main/experiments/Seal_fullsup $ROOT/ILNet-main/experiments/Seal_fullsup \
       $ROOT/SCTransNet-main/experiments/Seal_smoke $ROOT/ILNet-main/experiments/Seal_smoke

cd $ROOT/SCTransNet-main
nohup $PY train_seal.py --device cuda:0 --epochs 300 --val-interval 20 --batch-size 8 \
  --out experiments/Seal_fullsup > $ROOT/logs/sctransnet_train.log 2>&1 &
echo "sctransnet $!"

cd $ROOT/ILNet-main
nohup $PY train_seal.py --device cuda:0 --epochs 300 --val-interval 20 --batch-size 8 \
  --out experiments/Seal_fullsup > $ROOT/logs/ilnet_train.log 2>&1 &
echo "ilnet $!"
sleep 120
echo "== sctransnet"; tail -3 $ROOT/logs/sctransnet_train.log
echo "== ilnet"; tail -3 $ROOT/logs/ilnet_train.log
echo "== rpcanet"; grep val_iou $ROOT/logs/rpcanet_train.log | tail -2
echo "== rpcanetpp"; grep val_iou $ROOT/logs/rpcanetpp_train.log | tail -2
echo "== hdnet"; grep val_iou $ROOT/logs/hdnet_train.log | tail -2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
