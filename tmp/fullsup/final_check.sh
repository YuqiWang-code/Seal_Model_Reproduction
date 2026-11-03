#!/bin/bash
ROOT=/share_datasets/Seal_Model_Reproduction
echo "running sql jobs: $(pgrep -fc 'fct_sam_seal/bin/python train_seal' || true)"
for m in DNANet MSHNet UIU-Net; do
  echo "$m $(ls $ROOT/results/$m/Seal/fullsup_test_maps 2>/dev/null | wc -l) maps"
  python3 -c "
import json
d=json.load(open('$ROOT/results/$m/Seal/eight_metrics.json'))
print('  ', {k: round(v*100 if k!='Fa' and k!='MAE' else v, 8) for k,v in d['eight_metrics'].items()})
" 2>/dev/null || /home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -c "
import json
d=json.load(open('$ROOT/results/$m/Seal/eight_metrics.json'))
print('  ', d['eight_metrics'])
"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
