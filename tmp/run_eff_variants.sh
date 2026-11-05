#!/usr/bin/env bash
# 同一 harness 下顺带测我们自己的两个变体，供消融表参考：
#   --no-lora  : B + Fusion（原始 SAM3 单路共用，无模态 LoRA、无 LoRA3）
#   --no-lora3 : CM-LoRA + BISF（两路模态 LoRA，不挂共享 LoRA3）
set -euo pipefail
ROOT=/share_datasets/Seal_Model_Reproduction
CODE=$ROOT/Ours/CODE
PY=$ROOT/Ours/venv_ours/bin/python
export CUDA_VISIBLE_DEVICES=1

cd "$CODE"
for variant in no-lora no-lora3; do
  echo "=== $(date '+%F %T') $variant ==="
  $PY efficiency.py --sam_checkpoint "$ROOT/Ours/weights/sam3.pt" --img_size 1008 \
    --warmup 10 --iters 50 --$variant
  echo
done
echo "=== $(date '+%F %T') done ==="
