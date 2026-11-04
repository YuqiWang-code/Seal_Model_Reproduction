#!/bin/bash
# 原始版最终推理（GPU0，用最终选定的 best3.pth）+ 八项指标
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/Original
MAPS=$OUT/infer/best3/test
mkdir -p $OUT

echo "=== best3.pth ==="; ls -la $OUT/best3.pth
echo "=== 训练结束情况 ==="; grep "Epoch:" $R/logs/ours_original_train.log | tail -3

cd $R/Ours/Original_Code
SEAL_CUDA=0 $PY -u test.py --mode single \
  --checkpoint $OUT/best3.pth \
  --test_data_dir $R/Seal_data/test \
  --resnet_ckpt $R/Ours/weights/resnet50-19c8e357.pth \
  --save_dir $OUT/infer 2>&1 | tail -3
echo "预测张数: $(ls $MAPS/*.png | wc -l)"

cd $R
$PY evaluate.py --pred_root $MAPS --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $MAPS --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small.json 2>&1 | grep -A6 '"metrics"'

echo "=== 全监督版 val 曲线 ==="
grep "IoU:" $R/logs/ours_fullsup_train.log | tail -8
