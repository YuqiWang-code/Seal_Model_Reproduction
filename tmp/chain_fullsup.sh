#!/bin/bash
# 等两套模态 LoRA 训练完成（RGB 先、T 后），再在 GPU1 上启动全监督版训练。
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
LOG=$ROOT/logs/ours_fullsup_chain.log
: > $LOG

echo "$(date) 等待 T+LoRA/lora_weights.pt ..." >> $LOG
for i in $(seq 1 360); do   # 最多等 6 小时
  if [ -f "$ROOT/Ours/weights/T+LoRA/lora_weights.pt" ]; then
    if ! pgrep -f "train_lora.py --modality" > /dev/null; then
      echo "$(date) 两套 LoRA 就绪且训练进程已退出" >> $LOG
      break
    fi
  fi
  sleep 60
done

if [ ! -f "$ROOT/Ours/weights/T+LoRA/lora_weights.pt" ]; then
  echo "$(date) 超时：T+LoRA 仍未产出，放弃启动全监督训练" >> $LOG
  exit 1
fi

cd $ROOT/Ours/Fully_Supervised_Code_1104
echo "$(date) 启动全监督版训练 (GPU1)" >> $LOG
SEAL_CUDA=1 nohup $PY -u train.py --epoch 30 --lr 5e-5 --img_size 1008 \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --rgb_lora $ROOT/Ours/weights/RGB+LoRA/lora_weights.pt \
  --t_lora   $ROOT/Ours/weights/T+LoRA/lora_weights.pt \
  --train_data_dir $ROOT/Seal_data/train --val_data_dir $ROOT/Seal_data/val \
  --save_path $ROOT/results/Ours/FullySup \
  >> $ROOT/logs/ours_fullsup_train.log 2>&1 &
echo "$(date) fullsup pid=$!" >> $LOG
