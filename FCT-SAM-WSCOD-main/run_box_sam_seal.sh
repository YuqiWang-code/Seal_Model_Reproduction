#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
root=/share_datasets/Seal_Model_Reproduction
python=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
run="$root/results/FCT-SAM-WSCOD/Seal_box_sam"
cd "$root/FCT-SAM-WSCOD-main"
# Full-label quality review precedes this training entry.
"$python" -c 'import json,sys; q=json.load(open(sys.argv[1])); assert q["n_images"]==1500 and not q["invalid"] and q["all_boxes_nonempty"] and q["outside_box_pixels"]==0' "$run/quality_full/quality.json"
while true; do
  available=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  if (( available >= 15500 )); then break; fi
  echo "$(date --iso-8601=seconds) waiting for GPU 1: ${available} MiB free"
  sleep 60
done
"$python" -u train_seal.py --data "$root/Seal_data" --pseudo "$run/labels" \
  --init pretrain/Pvtv2_TE_new_dataset_batch4_epoch60.pth \
  --run "$run" --supervision box_sam --epochs 60 --size 768 "$@"
