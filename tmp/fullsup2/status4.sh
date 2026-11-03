#!/bin/bash
L=/share_datasets/Seal_Model_Reproduction/logs
for f in rpcanet2_train rpcanetpp3_train hdnet_train ilnet_train sctransnet_train; do
  echo "== $f"
  grep val_iou $L/$f.log | tail -2
  tail -1 $L/$f.log
done
date
