#!/bin/bash
# 全监督版最终推理 + 八项指标
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
OUT=$R/results/Ours/FullySup
MAPS=$OUT/test_maps

echo "=== best.pth ==="; ls -la $OUT/best.pth
echo "=== val 最优 ==="; grep 'IoU:' $R/logs/ours_fullsup_train.log | tail -2
echo "=== 训练是否结束 ==="; tail -1 $R/logs/ours_fullsup_train.log

cd $R/Ours/Fully_Supervised_Code_1104
SEAL_CUDA=1 $PY -u test_seal.py --weights $OUT/best.pth \
  --checkpoint $R/Ours/weights/sam3.pt \
  --rgb_lora $R/Ours/weights/RGB+LoRA/lora_weights.pt \
  --t_lora $R/Ours/weights/T+LoRA/lora_weights.pt \
  --test_data_dir $R/Seal_data/test --out_dir $MAPS 2>&1 | tail -3
echo "预测张数: $(ls $MAPS/*.png | wc -l)"

cd $R
$PY evaluate.py --pred_root $MAPS --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root $MAPS --gt_dir $R/Seal_data/test/GT \
  --json_output $OUT/metrics_small.json 2>&1 | grep -A6 '"metrics"'
