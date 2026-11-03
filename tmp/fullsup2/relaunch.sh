#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
mkdir -p $ROOT/logs

# HDNet: restart with a 120-epoch cosine schedule (300 epochs would not finish in budget)
PID=$(pgrep -f 'HDNet-main' | head -1)
[ -n "${PID:-}" ] && kill -9 $PID && echo "killed hdnet $PID"
sleep 3
rm -rf $ROOT/HDNet-main/experiments/Seal_fullsup
cd $ROOT/HDNet-main
nohup $PY train_seal.py --device cuda:1 --epochs 120 --val-interval 10 --out experiments/Seal_fullsup > $ROOT/logs/hdnet_train.log 2>&1 &
echo "hdnet $!"

# DHiF: dataset-level SoftIoU, 100 epochs
rm -rf $ROOT/DHiF-main/experiments/Seal_fullsup
cd $ROOT/DHiF-main
nohup $PY train_seal.py --device cuda:1 --epochs 100 --val-interval 10 --batch-size 8 \
  --out experiments/Seal_fullsup > $ROOT/logs/dhif_train.log 2>&1 &
echo "dhif $!"

# RPCANet++: dataset-level SoftIoU, 50 epochs
rm -rf $ROOT/RPCANet-main/experiments/Seal_fullsup_rpcanetpp2
cd $ROOT/RPCANet-main
nohup $PY train_seal.py --device cuda:1 --net-name rpcanet_pp --epochs 50 --val-interval 5 \
  --out experiments/Seal_fullsup_rpcanetpp2 > $ROOT/logs/rpcanetpp2_train.log 2>&1 &
echo "rpcanetpp2 $!"
sleep 120
for f in hdnet_train dhif_train rpcanetpp2_train; do echo "== $f"; tail -2 $ROOT/logs/$f.log; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
