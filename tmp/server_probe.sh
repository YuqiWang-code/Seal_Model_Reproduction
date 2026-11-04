#!/bin/bash
PY=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
$PY -V
$PY -c 'import torch, torchvision; print("torch", torch.__version__, "tv", torchvision.__version__, "cuda", torch.version.cuda)'
echo "--- pip conf ---"
cat ~/.pip/pip.conf 2>/dev/null
cat ~/.config/pip/pip.conf 2>/dev/null
$PY -m pip config list 2>/dev/null
echo "--- pip download test ---"
timeout 40 $PY -m pip download --no-deps -d /tmp/piptest iopath 2>&1 | tail -4
echo "--- installed? ---"
$PY -c 'import iopath' 2>&1 | tail -1
echo "--- sam3 imports ---"
grep -rhE '^(import|from) ' /share_datasets/Seal_Model_Reproduction/Ours/Original_Code/sam3 --include='*.py' | awk '{print $2}' | cut -d. -f1 | sort -u | tr '\n' ' '
echo
