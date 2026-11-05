#!/bin/bash
# BISF 残差化后重训 60 epoch
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
mkdir -p $R/results/Ours/BISF_res $R/logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd $R/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train.py --epoch 60 \
  --checkpoint $R/Ours/weights/sam3.pt \
  --train_data_dir $R/Seal_data/train --val_data_dir $R/Seal_data/val \
  --save_path $R/results/Ours/BISF_res \
  > $R/logs/ours_bisf_res_train.log 2>&1 &
echo "pid=$!"
sleep 120
tail -2 $R/logs/ours_bisf_res_train.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
