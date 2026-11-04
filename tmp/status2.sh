#!/bin/bash
R=/share_datasets/Seal_Model_Reproduction
echo "--- infer dryrun log ---"
tail -3 $R/logs/ours_original_infer_dryrun.log 2>/dev/null
echo "--- predictions ---"
ls $R/results/Ours/Original/infer_dryrun/best3/test 2>/dev/null | wc -l
echo "--- fullsup ---"
tail -2 $R/logs/ours_fullsup_train.log
grep -E "#VAL#" $R/logs/ours_fullsup_train.log | tail -3
echo "--- original ---"
grep "Epoch:" $R/logs/ours_original_train.log | tail -2
echo "--- gpu ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
