#!/bin/bash
# B+BISF 最终推理 + 八项指标（根目录统一入口）+ evaluate3 交叉核对
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/BISF
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

echo "=== best.pth ==="; ls -la $OUT/best.pth
echo "=== 训练结论 ==="; grep "IoU:" $R/logs/ours_bisf_train.log | tail -2

echo "=== 推理 Seal test 1983 张 ==="
$PY -u test.py --checkpoint $OUT/best.pth \
  --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir $R/Seal_data/test --save_dir $OUT/test_maps 2>&1 | tail -3
echo "预测张数: $(ls $OUT/test_maps/*.png | wc -l)"

cd $R
echo "=== 四项原函数 ==="
$PY evaluate.py --pred_root $OUT/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4.json 2>&1 | tail -8
echo "=== 小目标四项 ==="
$PY evaluate_small_targets.py --pred_root $OUT/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small.json 2>&1 | grep -A6 '"metrics"'
echo "=== evaluate3 交叉核对 ==="
cd $R/Ours/Fully_Supervised_Code_1104
$PY evaluate3.py --pred_root $OUT/test_maps --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/evaluate3.json 2>&1 | tail -12
