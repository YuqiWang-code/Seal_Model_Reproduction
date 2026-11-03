#!/bin/bash
L=/share_datasets/Seal_Model_Reproduction/logs
for f in hdnet_train ilnet_train sctransnet_train rpcanetpp_train rpcanet_test1; do
  echo "== $f"
  grep val_iou $L/$f.log | tail -2
  tail -1 $L/$f.log
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
date
