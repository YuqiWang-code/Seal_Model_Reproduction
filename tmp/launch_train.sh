#!/bin/bash
# 启动两个训练：GPU1 上的 RGB 模态 LoRA，GPU0 上的原始版全量训练
set -e
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
mkdir -p $ROOT/logs $ROOT/results/Ours/Original $ROOT/Ours/weights/RGB+LoRA

echo "--- GPU before ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "--- 启动 RGB LoRA (GPU1) ---"
cd $ROOT/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train_lora.py --modality rgb --epochs 8 --val_interval 2 \
  --out $ROOT/Ours/weights/RGB+LoRA \
  > $ROOT/logs/lora_rgb.log 2>&1 &
echo "pid=$!"

echo "--- 启动原始版训练 (GPU0) ---"
cd $ROOT/Ours/Original_Code
SEAL_CUDA=0 nohup $PY -u train.py --epoch 30 --lr 5e-5 --img_size 1008 --batchsize 1 \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --resnet_ckpt $ROOT/Ours/weights/resnet50-19c8e357.pth \
  --train_data_dir $ROOT/Seal_data/train --val_data_dir $ROOT/Seal_data/val \
  --save_path $ROOT/results/Ours/Original --save_name best3.pth \
  > $ROOT/logs/ours_original_train.log 2>&1 &
echo "pid=$!"

sleep 90
echo "--- GPU after 90s ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- lora log ---"; tail -3 $ROOT/logs/lora_rgb.log
echo "--- original log ---"; tail -3 $ROOT/logs/ours_original_train.log
