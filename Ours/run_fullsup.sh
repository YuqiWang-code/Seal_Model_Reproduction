#!/bin/bash
# 全监督 Ours（RGB-T 全像素 GT）在 Seal 上的训练 + 测试 + 八项指标
# 用法: bash Ours/run_fullsup.sh [epoch] [cuda]
set -e
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
EPOCH=${1:-30}
CUDA=${2:-1}
export SEAL_CUDA=$CUDA
W=$ROOT/Ours/weights
OUT=$ROOT/results/Ours/FullySup

for f in "$W/sam3.pt" "$W/RGB+LoRA/lora_weights.pt" "$W/T+LoRA/lora_weights.pt"; do
  [ -f "$f" ] || { echo "缺少权重: $f"; exit 1; }
done

mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/Ours/Fully_Supervised_Code_1104"

echo "== 训练 fullsup: epoch=$EPOCH cuda=$CUDA"
$PY train.py --epoch "$EPOCH" --lr 5e-5 --img_size 1008 \
  --checkpoint "$W/sam3.pt" \
  --rgb_lora "$W/RGB+LoRA/lora_weights.pt" --t_lora "$W/T+LoRA/lora_weights.pt" \
  --train_data_dir "$ROOT/Seal_data/train" --val_data_dir "$ROOT/Seal_data/val" \
  --save_path "$OUT" 2>&1 | tee "$ROOT/logs/ours_fullsup_train.log"

echo "== 推理 fullsup test"
$PY test_seal.py --weights "$OUT/best.pth" --checkpoint "$W/sam3.pt" \
  --rgb_lora "$W/RGB+LoRA/lora_weights.pt" --t_lora "$W/T+LoRA/lora_weights.pt" \
  --test_data_dir "$ROOT/Seal_data/test" --out_dir "$OUT/test_maps" \
  2>&1 | tee "$ROOT/logs/ours_fullsup_infer.log"

MAPS="$OUT/test_maps"
echo "预测张数: $(ls "$MAPS"/*.png | wc -l)"

cd "$ROOT"
echo "== 四项原函数"
$PY evaluate.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics4.json" 2>&1 | tail -8
echo "== 小目标四项"
$PY evaluate_small_targets.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics_small.json" 2>&1 | tail -8
