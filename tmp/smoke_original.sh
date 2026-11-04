#!/bin/bash
set -e
VENV=/share_datasets/Seal_Model_Reproduction/Ours/venv_ours
PY=$VENV/bin/python
ROOT=/share_datasets/Seal_Model_Reproduction/Ours
mkdir -p $ROOT/weights
echo "--- resnet50 download ---"
$PY - <<'PY'
import shutil, torchvision
from pathlib import Path
m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-19c8e357.pth"
dst = Path("/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth")
print("hub:", hub, hub.is_file())
if hub.is_file():
    shutil.copyfile(hub, dst)
    print("copied ->", dst, f"{dst.stat().st_size/1e6:.1f} MB")
PY
echo "--- smoke: original Model forward ---"
cd /share_datasets/Seal_Model_Reproduction/Ours/Original_Code
SEAL_CUDA=1 $PY - <<'PY'
import torch
from sam3.sam3model import Model
from pathlib import Path
m = Model(ckpt_path=None, image_size=1008, t_encoder_type="resnet50",
          resnet_ckpt="/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth")
m = m.eval().cuda()
rgb = torch.zeros(1, 3, 1008, 1008, device="cuda")
t = torch.zeros(1, 3, 1008, 1008, device="cuda")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    embed, hires = m.encode_images(rgb, t)
    print("image_embed", tuple(embed.shape), "high_res", [tuple(h.shape) for h in hires])
    mask1, _ = m.decode_noprompt(embed, hires)
    boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device="cuda")
    mask2, _ = m.decode_prompt(embed, hires, box_prompts=boxes)
    print("mask1", tuple(mask1.shape), "mask2", tuple(mask2.shape))
print("original smoke OK")
PY
