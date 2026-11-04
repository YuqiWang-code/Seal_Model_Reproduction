"""冒烟测试：SAM3 + LoRA 注入 + M1 前向（不需要任何权重）。

注意：SAM3 的 RoPE 表按 build 时的 image_size 预计算（默认 1008 -> 72x72），
所以这里必须用 1008 的输入，不能随意换分辨率。
"""

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SAM3_LoRA"))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from lora_layers import LoRAConfig, apply_lora_to_model, LoRALinear  # noqa: E402

BPE = HERE / "SAM3_LoRA" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
assert BPE.is_file(), BPE

device = torch.device("cuda")
print("building sam3 (random init, no checkpoint) ...", flush=True)
t0 = time.time()
model = build_sam3_image_model(
    bpe_path=str(BPE),
    device="cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)
print(f"built in {time.time() - t0:.1f}s, params(M) {sum(p.numel() for p in model.parameters()) / 1e6:.1f}", flush=True)

cfg = LoRAConfig(
    rank=16,
    alpha=32.0,
    dropout=0.0,
    target_modules=["qkv", "proj"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=False,
)
model = apply_lora_to_model(model, cfg)
sites = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
print("LoRA sites:", len(sites))
print("first:", sites[0][0], sites[0][1].in_features, sites[0][1].out_features)
print("last :", sites[-1][0], sites[-1][1].in_features, sites[-1][1].out_features)

from train import M1, SharedLoRA  # noqa: E402

named_shapes = {n: (m.in_features, m.out_features) for n, m in sites}
shared = SharedLoRA(named_shapes)
for name, module in sites:
    object.__setattr__(module, "shared_lora", shared)
    object.__setattr__(module, "shared_name", name)
print("shared LoRA3 params(M):", sum(p.numel() for p in shared.parameters()) / 1e6)

model.eval().to(device)
shared.to(device)
m1 = M1().to(device).eval()

x = torch.zeros(1, 3, 1008, 1008, device=device)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    t0 = time.time()
    out = model.backbone.forward_image(x)
    torch.cuda.synchronize()
    print(f"forward_image {time.time() - t0:.2f}s")
    print("keys:", sorted(out.keys()))
    print("vision_features:", tuple(out["vision_features"].shape))
    print("backbone_fpn:", [tuple(t.shape) for t in out["backbone_fpn"]])
    print("pos_enc:", [tuple(t.shape) for t in out["vision_pos_enc"]])
    dicts = {
        "backbone_fpn": out["backbone_fpn"],
        "vision_pos_enc": out["vision_pos_enc"],
        "vision_features": out["vision_features"],
    }
    fused = m1(dicts, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dicts.items()})
    print("fused fpn[-1]:", tuple(fused["backbone_fpn"][-1].shape))
    print("fused vision_features:", tuple(fused["vision_features"].shape))
print("smoke OK")
