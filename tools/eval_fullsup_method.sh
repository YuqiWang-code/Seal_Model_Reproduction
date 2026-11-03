#!/bin/bash
# Usage: eval_method.sh <method-dir> <result-name> <gpu>
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
DIR=$1; NAME=$2; GPU=${3:-1}
PRED=$ROOT/results/$NAME/Seal/fullsup_test_maps
mkdir -p $ROOT/results/$NAME/Seal
cd $ROOT/$DIR
echo "== inference $NAME on cuda:$GPU"
$PY infer_seal.py --device cuda:$GPU --out_dir $PRED > $ROOT/logs/infer_$NAME.log 2>&1 || { tail -20 $ROOT/logs/infer_$NAME.log; exit 1; }
tail -1 $ROOT/logs/infer_$NAME.log
echo "== count $(ls $PRED/*.png | wc -l)"
cd $ROOT
$PY evaluate.py --pred_root $PRED --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $ROOT/results/$NAME/Seal/fullsup_metrics.json > $ROOT/logs/eval4_$NAME.log 2>&1
$PY evaluate_small_targets.py --pred_root $PRED --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $ROOT/results/$NAME/Seal/fullsup_small_metrics.json > $ROOT/logs/evalsmall_$NAME.log 2>&1
echo "== four-metric"; tail -8 $ROOT/logs/eval4_$NAME.log
echo "== small-target"; grep -A6 '"metrics"' $ROOT/results/$NAME/Seal/fullsup_small_metrics.json | head -8
