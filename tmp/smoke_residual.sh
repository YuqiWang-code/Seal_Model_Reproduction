#!/bin/bash
# 冒烟：验证“残差化 + fuse 零初始化”后，未训练的 M1 与 B（--no-m1）预测逐像素一致
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

$PY - <<'PY'
import sys
from pathlib import Path
import torch
sys.path.insert(0, "/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104")
sys.path.insert(0, "/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/SAM3_LoRA")
from train import M1
torch.manual_seed(0)
m1 = M1()
state = m1.state_dict()
torch.save({"m1": state, "lora3": None, "epoch": 0, "iou": -1.0}, "/tmp/rand_m1.pth")
fuse_w = state["fuse.weight"].abs().max().item()
print("saved /tmp/rand_m1.pth; fuse|w|max =", fuse_w)
PY

rm -rf /tmp/mm_maps /tmp/b_maps
$PY -u test.py --checkpoint /tmp/rand_m1.pth --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir /tmp/seal_mini/test --save_dir /tmp/mm_maps 2>&1 | tail -2
$PY -u test.py --no-m1 --checkpoint /tmp/rand_m1.pth --sam_checkpoint $R/Ours/weights/sam3.pt \
  --test_data_dir /tmp/seal_mini/test --save_dir /tmp/b_maps 2>&1 | tail -2

$PY - <<'PY'
from pathlib import Path
import numpy as np, cv2
a = Path("/tmp/mm_maps"); b = Path("/tmp/b_maps")
names = sorted(p.name for p in a.glob("*.png"))
print("files:", len(names))
maxdiff = 0; same = 0
for n in names:
    x = cv2.imread(str(a / n), cv2.IMREAD_GRAYSCALE).astype(int)
    y = cv2.imread(str(b / n), cv2.IMREAD_GRAYSCALE).astype(int)
    d = int(np.abs(x - y).max())
    maxdiff = max(maxdiff, d)
    if d == 0:
        same += 1
print(f"identical={same}/{len(names)} maxdiff={maxdiff}")
PY
