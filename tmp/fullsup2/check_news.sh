#!/bin/bash
L=/share_datasets/Seal_Model_Reproduction/logs
for f in rpcanet_train rpcanetpp_train hdnet_train sctransnet_train ilnet_train; do
  echo "== $f"
  grep val_iou $L/$f.log | tail -2
  tail -1 $L/$f.log
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
date
