#!/bin/bash
echo "CVD=${CUDA_VISIBLE_DEVICES:-unset}"
env | grep -i -E "cuda|nvidia" || true
/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python -c "import torch; print('count', torch.cuda.device_count()); print('names', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
