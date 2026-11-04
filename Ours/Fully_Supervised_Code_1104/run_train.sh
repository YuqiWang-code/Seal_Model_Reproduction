#!/usr/bin/env bash
# B+BISF：冻结原始 SAM3（不用 RGB/T LoRA、不用 LoRA3），只训练 M1（即 BISF 模块）。
set -euo pipefail
cd "$(dirname "$0")"

ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
export SEAL_CUDA=${SEAL_CUDA:-1}

$PY train.py \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --train_data_dir $ROOT/Seal_data/train \
  --val_data_dir $ROOT/Seal_data/val \
  --save_path $ROOT/results/Ours/BISF \
  "$@"
