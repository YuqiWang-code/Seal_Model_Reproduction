#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -n 1)
export CUDA_VISIBLE_DEVICES=${GPU%%,*}

PYTHON=/home/yqwang/miniforge3/envs/lrdnet_seal/bin/python
PROJECT=/share_datasets/Seal_Model_Reproduction
DATA=$PROJECT/Seal_data
SOURCE=data_original
EXPERIMENT=experiments/SCOD
IMAGES=$SOURCE/TrainDataset/Imgs
SCRIBBLES=$SOURCE/Scribble

mkdir -p "$EXPERIMENT"
"$PYTHON" -u train.py --stage 1 --epochs 100 --batch_size 8 --validate_epoch 0 \
  --experiment_root "$EXPERIMENT" --train_image_root "$IMAGES" \
  --train_label_root "$SCRIBBLES" > "$EXPERIMENT/stage1_train.log" 2>&1

STAGE1=$EXPERIMENT/stage1/models/100_epoch.pth
"$PYTHON" generate_softgt.py --stage 1 --checkpoint "$STAGE1" \
  --experiment_root "$EXPERIMENT" --train_image_root "$IMAGES" \
  > "$EXPERIMENT/generate_softgt.log" 2>&1

"$PYTHON" -u train.py --stage 2 --epochs 100 --batch_size 8 --validate_epoch 0 \
  --checkpoint "$STAGE1" --experiment_root "$EXPERIMENT" \
  --train_image_root "$IMAGES" > "$EXPERIMENT/stage2_train.log" 2>&1

STAGE2=$EXPERIMENT/stage2/models/100_epoch.pth
for SOURCE_TEST in CAMO COD10K; do
  "$PYTHON" infer_seal.py --stage 2 --checkpoint "$STAGE2" \
    --experiment_root "$EXPERIMENT/$SOURCE_TEST" \
    --test_root "$SOURCE/TestDataset/$SOURCE_TEST" --test_image_dir Imgs \
    > "$EXPERIMENT/${SOURCE_TEST}_infer.log" 2>&1
  "$PYTHON" "$PROJECT/evaluate.py" \
    --pred_root "$PWD/$EXPERIMENT/$SOURCE_TEST/test_maps" \
    --gt_dir "$PWD/$SOURCE/TestDataset/$SOURCE_TEST/GT" \
    > "$EXPERIMENT/${SOURCE_TEST}_evaluate.log" 2>&1
done

"$PYTHON" infer_seal.py --stage 2 --checkpoint "$STAGE2" \
  --experiment_root "$EXPERIMENT" --data_root "$DATA" \
  > "$EXPERIMENT/Seal_infer.log" 2>&1
"$PYTHON" "$PROJECT/evaluate.py" \
  --pred_root "$PWD/$EXPERIMENT/test_maps" --gt_dir "$DATA/test/GT" \
  > "$EXPERIMENT/Seal_evaluate.log" 2>&1
