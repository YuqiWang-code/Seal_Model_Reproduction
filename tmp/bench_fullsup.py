"""全监督版训练步冒烟 + 速度/显存基准（随机初始化，不需要权重）。

复刻 train.py 的关键路径：SAM3(随机) + 模态 LoRA + 共享 LoRA3 + M1，
真实 Seal 样本上跑 3 个训练步（结构 + focal + dice），报告每步耗时与峰值显存。
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from lora_layers import LoRAConfig, apply_lora_to_model  # noqa: E402

from data import get_loader  # noqa: E402
from train import (  # noqa: E402
    M1,
    SharedLoRA,
    dice_loss,
    enable_trunk_checkpoint,
    encode_visual_text,
    forward_sample,
    make_find_stage,
    resize_gt,
    sigmoid_focal_loss,
    upsample_logits,
)
from utils import structure_loss  # noqa: E402

SEAL = "/share_datasets/Seal_Model_Reproduction/Seal_data"
BPE = HERE / "SAM3_LoRA" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEVICE = torch.device("cuda")
STEPS = 3

cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.0, target_modules=["qkv", "proj"],
                 apply_to_vision_encoder=True, apply_to_text_encoder=False,
                 apply_to_geometry_encoder=False, apply_to_detr_encoder=False,
                 apply_to_detr_decoder=False, apply_to_mask_decoder=False)


def build_model():
    model = build_sam3_image_model(
        bpe_path=str(BPE), device="cpu", eval_mode=True, checkpoint_path=None,
        load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False, compile=False,
    )
    return apply_lora_to_model(model, cfg)


print("building two SAM3 copies (RGB / T) ...", flush=True)
t0 = time.time()
rgb_model, t_model = build_model(), build_model()
for param in rgb_model.parameters():
    param.requires_grad = False
for param in t_model.parameters():
    param.requires_grad = False
rgb_model.eval().to(DEVICE)
t_model.eval().to(DEVICE)
print(f"built in {time.time() - t0:.1f}s", flush=True)

sites_rgb = {n: m for n, m in rgb_model.named_modules() if type(m).__name__ == "LoRALinear"}
sites_t = {n: m for n, m in t_model.named_modules() if type(m).__name__ == "LoRALinear"}
shared = SharedLoRA({n: (m.in_features, m.out_features) for n, m in sites_rgb.items()}).to(DEVICE)
for name, module in sites_rgb.items():
    object.__setattr__(module, "shared_lora", shared)
    object.__setattr__(module, "shared_name", name)
for name, module in sites_t.items():
    object.__setattr__(module, "shared_lora", shared)
    object.__setattr__(module, "shared_name", name)
enable_trunk_checkpoint(rgb_model)
enable_trunk_checkpoint(t_model)

m1 = M1().to(DEVICE).train()
params = [p for p in m1.parameters() if p.requires_grad] + [p for p in shared.parameters() if p.requires_grad]
print("trainable params(M):", sum(p.numel() for p in params) / 1e6, flush=True)
optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)

loader = get_loader(f"{SEAL}/train", image_size=1008, batch_size=1, shuffle=False,
                    num_workers=2, load_gt=True)
text_outputs = encode_visual_text(rgb_model, DEVICE)
find_stage = make_find_stage(DEVICE)

torch.cuda.reset_peak_memory_stats()
step = 0
for batch in loader:
    step += 1
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    rgb = batch["rgb"].unsqueeze(0).cuda(non_blocking=True)
    thermal = batch["t"].unsqueeze(0).cuda(non_blocking=True)
    gt_boxes = batch["boxes"].cuda(non_blocking=True)
    gt_mask = batch["gt_mask"].cuda(non_blocking=True)

    pred_logit = forward_sample(rgb, thermal, gt_boxes, rgb_model, t_model, m1,
                                text_outputs, find_stage, grad_backbone=True)
    pred_map = upsample_logits(pred_logit, rgb.shape[-2:])
    gt = resize_gt(gt_mask, rgb.shape[-2:])
    loss = structure_loss(pred_map, gt) + sigmoid_focal_loss(pred_map, gt) + dice_loss(pred_map, gt)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step {step}: {time.time() - t0:.2f}s loss={float(loss):.5f} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
    if step >= STEPS:
        break
print("fullsup train-step smoke OK")
