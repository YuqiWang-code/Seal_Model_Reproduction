#!/bin/bash
# 原始版 test.py 真实权重试跑（用当前 best3.pth，仅验证推理链路；训练结束后会用最终权重重跑）
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
cd $ROOT/Ours/Original_Code
SEAL_CUDA=1 nohup $PY -u test.py --mode single \
  --checkpoint $ROOT/results/Ours/Original/best3.pth \
  --test_data_dir $ROOT/Seal_data/test \
  --resnet_ckpt $ROOT/Ours/weights/resnet50-19c8e357.pth \
  --save_dir $ROOT/results/Ours/Original/infer_dryrun \
  > $ROOT/logs/ours_original_infer_dryrun.log 2>&1 &
echo "dryrun pid=$!"
