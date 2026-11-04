#!/bin/bash
# 用同一套流程把 B（--no-m1）在 Seal val 上跑一遍，直接和 B+BISF 的 val IoU 0.50111 对比
set -e
R=/share_datasets/Seal_Model_Reproduction
PY=$R/Ours/venv_ours/bin/python
export SEAL_CUDA=1
cd $R/Ours/Fully_Supervised_Code_1104

$PY - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104")))
sys.path.insert(0, "/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/SAM3_LoRA")
import torch
from data import get_loader
from train import (build_frozen_base_model, encode_visual_text, make_find_stage,
                   forward_sample, prediction_probability, iou_counts, dataset_iou)

device = torch.device("cuda")
model = build_frozen_base_model(Path("/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt"), device)
loader = get_loader("/share_datasets/Seal_Model_Reproduction/Seal_data/val", image_size=1008,
                    batch_size=1, shuffle=False, num_workers=0, load_gt=True)
text_outputs = encode_visual_text(model, device)
find_stage = make_find_stage(device)
tp = fp = fn = 0
n = 0
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for batch in loader:
        rgb = batch["rgb"].unsqueeze(0).cuda()
        thermal = batch["t"].unsqueeze(0).cuda()
        boxes = batch["boxes"].cuda()
        gt_mask = batch["gt_mask"].cuda()
        logits = forward_sample(rgb, thermal, boxes, model, model, None, text_outputs, find_stage)
        prob = prediction_probability(logits, gt_mask)
        a, b, c = iou_counts(prob, gt_mask)
        tp, fp, fn = tp + a, fp + b, fn + c
        n += 1
print(f"B (average fusion) val: n={n} tp={tp} fp={fp} fn={fn} IoU={dataset_iou(tp, fp, fn):.6f}")
PY
