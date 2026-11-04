#!/bin/bash
set -e
VENV=/share_datasets/Seal_Model_Reproduction/Ours/venv_ours
PY=$VENV/bin/python
ROOT=/share_datasets/Seal_Model_Reproduction/Ours
echo "--- hub cache ---"
ls -la ~/.cache/torch/hub/checkpoints/ | tail -5
SRC=$(ls -S ~/.cache/torch/hub/checkpoints/resnet50-*.pth | head -1)
echo "src: $SRC"
cp -f "$SRC" "$ROOT/weights/resnet50-19c8e357.pth"
ls -la "$ROOT/weights/"
echo "--- smoke: original Model forward ---"
cd $ROOT/Original_Code
SEAL_CUDA=1 $PY - <<'PY'
import torch
from sam3.sam3model import Model
m = Model(ckpt_path=None, image_size=1008, t_encoder_type="resnet50",
          resnet_ckpt="/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth")
m = m.eval().cuda()
print("params(M):", sum(p.numel() for p in m.parameters()) / 1e6,
      "trainable(M):", sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6)
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
