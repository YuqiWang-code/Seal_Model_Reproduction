"""诊断：全监督版随机初始化下 loss=nan 的来源（只跑 1 步）。"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from lora_layers import LoRAConfig, apply_lora_to_model  # noqa: E402

from data import get_loader  # noqa: E402
from train import (  # noqa: E402
    M1, SharedLoRA, branch_backbone, decode, dice_loss, encode_visual_text,
    make_find_stage, merge_topk_logits, resize_gt, sigmoid_focal_loss, upsample_logits,
)
from utils import structure_loss  # noqa: E402

SEAL = "/share_datasets/Seal_Model_Reproduction/Seal_data"
BPE = HERE / "SAM3_LoRA" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEVICE = torch.device("cuda")
cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.0, target_modules=["qkv", "proj"])


def stats(name, x):
    x = x.float()
    print(f"{name}: shape={tuple(x.shape)} min={x.min():.4g} max={x.max():.4g} "
          f"mean={x.mean():.4g} nan={int(torch.isnan(x).sum())} inf={int(torch.isinf(x).sum())}", flush=True)


def build():
    m = build_sam3_image_model(bpe_path=str(BPE), device="cpu", eval_mode=True, checkpoint_path=None,
                               load_from_HF=False, enable_segmentation=True,
                               enable_inst_interactivity=False, compile=False)
    return apply_lora_to_model(m, cfg)


rgb_model, t_model = build(), build()
for p in rgb_model.parameters():
    p.requires_grad = False
for p in t_model.parameters():
    p.requires_grad = False
rgb_model.eval().to(DEVICE)
t_model.eval().to(DEVICE)
sites = {n: m for n, m in rgb_model.named_modules() if type(m).__name__ == "LoRALinear"}
shared = SharedLoRA({n: (m.in_features, m.out_features) for n, m in sites.items()}).to(DEVICE)
for name, module in sites.items():
    object.__setattr__(module, "shared_lora", shared)
    object.__setattr__(module, "shared_name", name)
m1 = M1().to(DEVICE).eval()

loader = get_loader(f"{SEAL}/train", image_size=1008, batch_size=1, shuffle=False, num_workers=0, load_gt=True)
batch = next(iter(loader))
rgb = batch["rgb"].unsqueeze(0).cuda()
thermal = batch["t"].unsqueeze(0).cuda()
gt_boxes = batch["boxes"].cuda()
gt_mask = batch["gt_mask"].cuda()
print("boxes:", gt_boxes.tolist(), "gt fg:", int(gt_mask.sum()))

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    rgb_out = rgb_model.backbone.forward_image(rgb)
    t_out = t_model.backbone.forward_image(thermal)
    stats("rgb vision_features", rgb_out["vision_features"])
    stats("t   vision_features", t_out["vision_features"])
    fused = m1(rgb_out, t_out)
    stats("fused fpn[-1]", fused["backbone_fpn"][-1])
    text_outputs = encode_visual_text(rgb_model, DEVICE)
    for k, v in text_outputs.items():
        if torch.is_tensor(v):
            stats(f"text[{k}]", v)
        elif isinstance(v, (list, tuple)):
            for i, t in enumerate(v):
                if torch.is_tensor(t):
                    stats(f"text[{k}][{i}]", t)
    find_stage = make_find_stage(DEVICE)
    features = branch_backbone(fused, text_outputs)
    out = decode(rgb_model, features, gt_boxes, find_stage)
    for k in ("pred_logits", "pred_masks", "presence_logit_dec"):
        if k in out and torch.is_tensor(out[k]):
            stats(f"decoder {k}", out[k])
    merged = merge_topk_logits(out, int(gt_boxes.shape[0]))
    stats("merged logits", merged)
    pred_map = upsample_logits(merged, rgb.shape[-2:])
    stats("pred_map", pred_map)
    gt = resize_gt(gt_mask, rgb.shape[-2:])
    s = structure_loss(pred_map, gt)
    f = sigmoid_focal_loss(pred_map, gt)
    d = dice_loss(pred_map, gt)
    print(f"structure={float(s):.6f} focal={float(f):.6f} dice={float(d):.6f}", flush=True)
