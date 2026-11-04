#!/bin/bash
set -e
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
export SEAL_CUDA=1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo "=== lora smoke: rgb, 2 train images, 1 epoch ==="
cd $ROOT/Ours/Fully_Supervised_Code_1104
$PY train_lora.py --modality rgb --limit 2 --epochs 1 --val_interval 1 --num_workers 0 \
  --out /tmp/lora_smoke 2>&1 | tail -8
ls -la /tmp/lora_smoke/lora_weights.pt 2>/dev/null || echo "(no lora saved)"

echo "=== fullsup train.py smoke on mini (real sam3.pt + dummy lora) ==="
$PY train.py --epoch 1 --lr 5e-5 --img_size 1008 \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --rgb_lora $ROOT/Ours/weights/_dummy_lora/RGB+LoRA/lora_weights.pt \
  --t_lora  $ROOT/Ours/weights/_dummy_lora/T+LoRA/lora_weights.pt \
  --train_data_dir /tmp/seal_mini/train --val_data_dir /tmp/seal_mini/val \
  --save_path /tmp/fullsup_smoke 2>&1 | tail -5
