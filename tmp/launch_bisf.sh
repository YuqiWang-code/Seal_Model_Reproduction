#!/bin/bash
# 启动 B+BISF 正式训练（Seal train 1500 / val 500，30 epoch，GPU1）
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
mkdir -p $R/results/Ours/BISF $R/logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd $R/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train.py --epoch 30 \
  --checkpoint $R/Ours/weights/sam3.pt \
  --train_data_dir $R/Seal_data/train --val_data_dir $R/Seal_data/val \
  --save_path $R/results/Ours/BISF \
  > $R/logs/ours_bisf_train.log 2>&1 &
echo "pid=$!"
sleep 120
echo "--- log ---"; tail -3 $R/logs/ours_bisf_train.log
echo "--- gpu ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
