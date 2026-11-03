#!/bin/bash
L=/share_datasets/Seal_Model_Reproduction/logs
for f in hdnet_train ilnet_train sctransnet_train dhif_train rpcanetpp_train; do
  echo "== $f"
  grep val_iou $L/$f.log 2>/dev/null | tail -2
  tail -1 $L/$f.log 2>/dev/null
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
date
