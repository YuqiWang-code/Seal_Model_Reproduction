#!/bin/bash
set -e
BASE=/home/yqwang/miniforge3/envs/fct_sam_seal/bin/python
ROOT=/share_datasets/Seal_Model_Reproduction/Ours
VENV=$ROOT/venv_ours
if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv (system-site-packages) ..."
  $BASE -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip 2>/dev/null || true
"$VENV/bin/python" -m pip install --quiet \
  iopath hydra-core omegaconf einops ftfy regex huggingface_hub psutil 2>&1 | tail -3
"$VENV/bin/python" - <<'PY'
import torch, iopath, hydra, omegaconf, einops, ftfy, regex, huggingface_hub, psutil
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("deps OK")
PY
