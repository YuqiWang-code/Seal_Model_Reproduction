#!/bin/bash
# 等 RGB LoRA 训练进程结束后，接着在 GPU1 上训练 T 模态 LoRA
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
RGB_PID=$1
while kill -0 "$RGB_PID" 2>/dev/null; do
  sleep 60
done
echo "$(date) RGB LoRA 结束，开始 T LoRA" >> $ROOT/logs/lora_t_chain.log
cd $ROOT/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 nohup $PY -u train_lora.py --modality t --epochs 8 --val_interval 2 \
  --out $ROOT/Ours/weights/T+LoRA \
  >> $ROOT/logs/lora_t.log 2>&1 &
echo "started T lora pid=$!" >> $ROOT/logs/lora_t_chain.log
