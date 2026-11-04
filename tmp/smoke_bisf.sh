#!/bin/bash
# B+BISF 冒烟：迷你子集 1 epoch 训练 + 测试推理（验证新包在本环境可跑通）
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

rm -rf /tmp/bisf_smoke /tmp/bisf_smoke_maps
echo "=== train 1 epoch on mini (24/6) ==="
$PY train.py --epoch 1 --checkpoint $R/Ours/weights/sam3.pt \
  --train_data_dir /tmp/seal_mini/train --val_data_dir /tmp/seal_mini/val \
  --save_path /tmp/bisf_smoke --num_workers 0 2>&1 | tail -6
ls -la /tmp/bisf_smoke/best.pth

echo "=== test on mini (6 张) ==="
$PY test.py --checkpoint /tmp/bisf_smoke/best.pth \
  --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir /tmp/seal_mini/test --save_dir /tmp/bisf_smoke_maps 2>&1 | tail -4
ls /tmp/bisf_smoke_maps | wc -l

echo "=== evaluate3 on mini ==="
$PY evaluate3.py --pred_root /tmp/bisf_smoke_maps --gt_dir /tmp/seal_mini/test/GT 2>&1 | tail -12
