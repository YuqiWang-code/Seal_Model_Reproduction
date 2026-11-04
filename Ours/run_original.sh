#!/bin/bash
# 原始 Ours（RGB-T + 框提示双解码器）在 Seal 上的训练 + 测试 + 八项指标
# 用法: bash Ours/run_original.sh [epoch] [cuda]
set -e
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
EPOCH=${1:-30}
CUDA=${2:-1}
export SEAL_CUDA=$CUDA
W=$ROOT/Ours/weights
OUT=$ROOT/results/Ours/Original

for f in "$W/sam3.pt" "$W/resnet50-19c8e357.pth"; do
  [ -f "$f" ] || { echo "缺少权重: $f"; exit 1; }
done

mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/Ours/Original_Code"

echo "== 训练 original: epoch=$EPOCH cuda=$CUDA"
$PY train.py --epoch "$EPOCH" --lr 5e-5 --img_size 1008 \
  --checkpoint "$W/sam3.pt" --resnet_ckpt "$W/resnet50-19c8e357.pth" \
  --train_data_dir "$ROOT/Seal_data/train" --val_data_dir "$ROOT/Seal_data/val" \
  --save_path "$OUT" --save_name best3.pth 2>&1 | tee "$ROOT/logs/ours_original_train.log"

echo "== 推理 original test"
$PY test.py --mode single --checkpoint "$OUT/best3.pth" \
  --test_data_dir "$ROOT/Seal_data/test" --resnet_ckpt "$W/resnet50-19c8e357.pth" \
  --save_dir "$OUT/infer" 2>&1 | tee "$ROOT/logs/ours_original_infer.log"

MAPS="$OUT/infer/best3/test"
echo "预测张数: $(ls "$MAPS"/*.png | wc -l)"

cd "$ROOT"
echo "== 四项原函数"
$PY evaluate.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics4.json" 2>&1 | tail -8
echo "== 小目标四项"
$PY evaluate_small_targets.py --pred_root "$MAPS" --gt_dir "$ROOT/Seal_data/test/GT" \
  --json_output "$OUT/metrics_small.json" 2>&1 | tail -8
