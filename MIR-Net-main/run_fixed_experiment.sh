#!/usr/bin/env bash
set -euo pipefail
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
METHOD=$ROOT/MIR-Net-main
NAME=$1
GPU=${2:-$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n -k3,3n | head -n 1 | cut -d, -f1 | tr -d ' ')}
EXP=$METHOD/experiments/${NAME}_original_eval
OUT=$ROOT/results/MIR-Net/Seal/${NAME}_original_eval_test_maps
LABELS=$METHOD/experiments/Seal_box_sam/labels
ARGS=(--dense_labels --ce_input logits --epoch 60)
if [ "$NAME" = Seal ]; then
  LABELS=$METHOD/experiments/Seal/box_labels
  ARGS=(--ce_input probabilities --epoch 60)
elif [ "$NAME" = Seal_box_sam_150 ]; then
  ARGS=(--dense_labels --ce_input logits --epoch 150 --lr_decay_epoch 50)
fi
cd "$METHOD"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_seal.py \
  --data_root "$ROOT/Seal_data" --labels "$LABELS" \
  --initial_checkpoint "$METHOD/published_checkpoint/31_wM_test1/model-best.pth" \
  --output "$EXP" "${ARGS[@]}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u infer_seal.py \
  --data "$ROOT/Seal_data/test" --checkpoint "$EXP/model-best.pth" --output "$OUT"
"$PY" audit_fixed_evaluation.py --root "$ROOT" --reference "$EXP/evaluate_source.py" \
  --pred "$OUT" --gt_dir "$ROOT/Seal_data/test/GT" \
  --output "$EXP/test_metrics.json"
