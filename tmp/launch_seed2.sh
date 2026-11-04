#!/bin/bash
# 第二个随机种子（seed=2）稳定性对照：全监督版在 GPU1（与 seed1 共享），原始版在 GPU0
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
mkdir -p $R/results/Ours/FullySup_seed2 $R/results/Ours/Original_seed2

echo "--- GPU before ---"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

cd $R/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train.py --epoch 30 --lr 5e-5 --img_size 1008 --seed 2 \
  --checkpoint $R/Ours/weights/sam3.pt \
  --rgb_lora $R/Ours/weights/RGB+LoRA/lora_weights.pt \
  --t_lora   $R/Ours/weights/T+LoRA/lora_weights.pt \
  --train_data_dir $R/Seal_data/train --val_data_dir $R/Seal_data/val \
  --save_path $R/results/Ours/FullySup_seed2 \
  > $R/logs/ours_fullsup_seed2.log 2>&1 &
echo "fullsup seed2 pid=$!"

cd $R/Ours/Original_Code
SEAL_CUDA=0 nohup $PY -u train.py --epoch 30 --lr 5e-5 --img_size 1008 --batchsize 1 --seed 2 \
  --checkpoint $R/Ours/weights/sam3.pt \
  --resnet_ckpt $R/Ours/weights/resnet50-19c8e357.pth \
  --train_data_dir $R/Seal_data/train --val_data_dir $R/Seal_data/val \
  --save_path $R/results/Ours/Original_seed2 --save_name best3.pth \
  > $R/logs/ours_original_seed2.log 2>&1 &
echo "original seed2 pid=$!"

sleep 120
echo "--- GPU after 120s ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "--- fullsup seed2 ---"; tail -2 $R/logs/ours_fullsup_seed2.log
echo "--- original seed2 ---"; tail -2 $R/logs/ours_original_seed2.log
echo "--- seed1 fullsup ---"; grep 'IoU:' $R/logs/ours_fullsup_train.log | tail -1
