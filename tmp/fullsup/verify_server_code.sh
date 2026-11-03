#!/bin/bash
set -u
ROOT=/share_datasets/Seal_Model_Reproduction
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
mkdir -p $ROOT/tools
for d in ACM-main DNANet-main MSHNet-main UIU-Net-main; do
  echo "== $d"
  cd $ROOT/$d
  $PY -m py_compile train_seal.py infer_seal.py seal_dataset.py seal_infer.py && echo "compile ok"
done
cd $ROOT
$PY -m py_compile tools/merge_seal_metrics.py tools/prepare_seal_irstd.py && echo "tools ok"
echo "== imports"
cd $ROOT/DNANet-main && $PY -c "import seal_dataset, seal_infer, torch; print('dnanet import ok')"
