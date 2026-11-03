#!/bin/bash
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
echo "== form A (bash -s script) =="
$PY -c "import torch;print('count',torch.cuda.device_count())"
echo "== form B (timeout wrapper) =="
timeout 60 $PY -c "import torch;print('count',torch.cuda.device_count())"
echo "== form C (bash -lc) =="
bash -lc "$PY -c \"import torch;print('count',torch.cuda.device_count())\""
echo "== nvidia-smi =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
