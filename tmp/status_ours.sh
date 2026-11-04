#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== LoRA RGB ==="
grep -E 'VAL|TRAIN' $R/logs/lora_rgb.log 2>/dev/null | tail -4
tail -1 $R/logs/lora_rgb.log 2>/dev/null
echo "=== LoRA T ==="
grep -E 'VAL|TRAIN' $R/logs/lora_t.log 2>/dev/null | tail -4
tail -1 $R/logs/lora_t.log 2>/dev/null
cat $R/logs/lora_t_chain.log 2>/dev/null
echo "=== Original ==="
grep -E 'Epoch:' $R/logs/ours_original_train.log 2>/dev/null | tail -3
tail -1 $R/logs/ours_original_train.log 2>/dev/null
echo "=== weights ==="
ls -la $R/Ours/weights/RGB+LoRA/ $R/Ours/weights/T+LoRA/ 2>/dev/null
