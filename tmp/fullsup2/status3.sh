#!/bin/bash
L=/share_datasets/Seal_Model_Reproduction/logs
for f in ilnet_train sctransnet_train hdnet_train dhif_msh_train rpcanetpp2_train dhif_train; do
  echo "== $f"
  grep val_iou $L/$f.log 2>/dev/null | tail -2
  tail -1 $L/$f.log 2>/dev/null
done
date
