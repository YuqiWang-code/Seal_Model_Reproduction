#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

# stop the diverging RPCANet++ run (peaked at val IoU 0.298 on epoch 4, then diverged)
PID=$(pgrep -f 'rpcanet_pp --epochs 50' | head -1)
[ -n "${PID:-}" ] && kill -9 $PID && echo "killed rpcanetpp2 $PID"
sleep 3
rm -rf $ROOT/RPCANet-main/experiments/Seal_fullsup_rpcanetpp3
cd $ROOT/RPCANet-main
nohup $PY train_seal.py --device cuda:1 --net-name rpcanet_pp --epochs 60 --val-interval 2 \
  --lr 1e-4 --out experiments/Seal_fullsup_rpcanetpp3 > $ROOT/logs/rpcanetpp3_train.log 2>&1 &
echo "rpcanetpp3 $!"

rm -rf $ROOT/DHiF-main/experiments/Seal_msh
cd $ROOT/DHiF-main
nohup $PY train_seal.py --device cuda:1 --model MSH --epochs 100 --val-interval 10 --batch-size 8 \
  --out experiments/Seal_msh > $ROOT/logs/dhif_msh_train.log 2>&1 &
echo "dhif_msh $!"
sleep 120
echo "== rpcanetpp3"; tail -3 $ROOT/logs/rpcanetpp3_train.log
echo "== dhif_msh"; tail -3 $ROOT/logs/dhif_msh_train.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
