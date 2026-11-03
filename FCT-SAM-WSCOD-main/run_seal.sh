#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
root=/share_datasets/Seal_Model_Reproduction
python=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
cd "$root/FCT-SAM-WSCOD-main"
"$python" download_sam.py
"$python" prepare_seal.py --data "$root/Seal_data" \
  --sam pretrained/sam_vit_h_4b8939.pth --output "$root/Seal_data/train/SAM_1fg5bg"
"$python" train_seal.py --data "$root/Seal_data" --pseudo "$root/Seal_data/train/SAM_1fg5bg" \
  --init pretrain/Pvtv2_TE_new_dataset_batch4_epoch60.pth \
  --run "$root/results/FCT-SAM-WSCOD/Seal_retrain_768_1fg5bg" --epochs 60 --size 768
