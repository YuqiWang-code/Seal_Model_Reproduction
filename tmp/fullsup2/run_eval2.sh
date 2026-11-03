#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
DIR=${DIR:?}
NAME=${NAME:?}
GPU=${GPU:-0}
WEIGHTS=${WEIGHTS:-experiments/Seal_fullsup/best.pth}
EXTRA=${EXTRA:-}
PRED=$ROOT/results/$NAME/Seal/fullsup_test_maps
mkdir -p $ROOT/results/$NAME/Seal

cd $ROOT/$DIR
$PY infer_seal.py --device cuda:$GPU --weights "$WEIGHTS" $EXTRA --out_dir $PRED > $ROOT/logs/infer_$NAME.log 2>&1
rc=$?
echo "infer rc=$rc count=$(ls $PRED/*.png 2>/dev/null | wc -l)"
tail -1 $ROOT/logs/infer_$NAME.log
[ $rc -ne 0 ] && exit 1

cd $ROOT
$PY evaluate.py --pred_root $PRED --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $ROOT/results/$NAME/Seal/fullsup_metrics.json > $ROOT/logs/eval4_$NAME.log 2>&1
cat $ROOT/results/$NAME/Seal/fullsup_metrics.json

$PY evaluate_small_targets.py --pred_root $PRED --gt_dir $ROOT/Seal_data/test/GT \
  --json_output $ROOT/results/$NAME/Seal/fullsup_small_metrics.json > $ROOT/logs/evalsmall_$NAME.log 2>&1
$PY $ROOT/tools/merge_seal_metrics.py $NAME $ROOT/results/$NAME/Seal
