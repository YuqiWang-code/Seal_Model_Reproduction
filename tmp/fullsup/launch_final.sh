#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python

# detach MSHNet from its chain (keep the running python, drop the queued UIU-Net/ACM steps)
pkill -9 -f gpu1_chain2.sh || true
sleep 2
pgrep -af 'train_seal.py --device cuda:1' | head -2

# ACM joins GPU 1 now: it is a small model and would otherwise wait ~6 h
rm -rf $ROOT/ACM-main/experiments/Seal_fullsup
cd $ROOT/ACM-main
nohup $PY train_seal.py --device cuda:1 --epochs 300 --val-interval 20 --out experiments/Seal_fullsup > $ROOT/logs/acm_train.log 2>&1 &
echo "acm pid $!"

# UIU-Net starts as soon as DNA-Net writes its final checkpoint
cat > $ROOT/logs/wait_uiunet.sh <<'EOF'
#!/bin/bash
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
while [ ! -f $ROOT/DNANet-main/experiments/Seal_fullsup/last.pth ]; do sleep 120; done
sleep 30
rm -rf $ROOT/UIU-Net-main/experiments/Seal_fullsup
cd $ROOT/UIU-Net-main
$PY train_seal.py --device cuda:1 --epochs 500 --val-interval 20 --out experiments/Seal_fullsup > $ROOT/logs/uiunet_train.log 2>&1
echo "UIU-Net done $(date)" >> $ROOT/logs/chain.log
EOF
nohup bash $ROOT/logs/wait_uiunet.sh > $ROOT/logs/wait_uiunet.out 2>&1 &
echo "waiter pid $!"
sleep 45
tail -1 $ROOT/logs/acm_train.log; tail -1 $ROOT/logs/mshnet_train.log; tail -1 $ROOT/logs/dnanet_train.log
