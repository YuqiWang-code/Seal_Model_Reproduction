#!/bin/bash
# 30 epoch 的 B+BISF 未全面超过表里 B 行（Fw/Pd/S 略低），追加 60 epoch（lr 在第 50 epoch 衰减）
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
mkdir -p $R/results/Ours/BISF_60ep $R/logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd $R/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train.py --epoch 60 \
  --checkpoint $R/Ours/weights/sam3.pt \
  --train_data_dir $R/Seal_data/train --val_data_dir $R/Seal_data/val \
  --save_path $R/results/Ours/BISF_60ep \
  > $R/logs/ours_bisf_60ep.log 2>&1 &
echo "pid=$!"
sleep 60
tail -2 $R/logs/ours_bisf_60ep.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
