#!/usr/bin/env bash
# 在 RTX 5090（GPU1）上按 Ours/CODE/efficiency.py 实测 Params / GFLOPs / FPS。
# 输入 1008×1008（SAM3 rope 表固定 1008=72×14，不支持其它分辨率）。
set -euo pipefail
ROOT=/share_datasets/Seal_Model_Reproduction
CODE=$ROOT/Ours/CODE
PY=$ROOT/Ours/venv_ours/bin/python
export CUDA_VISIBLE_DEVICES=1

cd "$CODE"
echo "=== $(date '+%F %T') 1008x1008 ==="
$PY efficiency.py \
  --sam_checkpoint "$ROOT/Ours/weights/sam3.pt" \
  --img_size 1008 \
  --warmup 10 \
  --iters 50
echo "=== $(date '+%F %T') done ==="
