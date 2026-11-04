"""验证重写的 load_lora_weights：保存→加载→前向一致。

A、B 两套模型的**基础权重完全相同**（B 先装入 A 的非 LoRA 参数），只把 B 的 LoRA 置零，
再用 load_lora_weights 装回 A 的 LoRA，比较 ``backbone.forward_image`` 输出。
三种前缀（原样 / module. / backbone.）都要能匹配 128 个张量。
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/SAM3_LoRA")

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights  # noqa: E402

BPE = Path("/share_datasets/Seal_Model_Reproduction/Ours/Fully_Supervised_Code_1104/SAM3_LoRA/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.0, target_modules=["qkv", "proj"])


def build():
    model = build_sam3_image_model(bpe_path=str(BPE), device="cpu", eval_mode=True,
                                   checkpoint_path=None, load_from_HF=False,
                                   enable_segmentation=True, enable_inst_interactivity=False,
                                   compile=False)
    return apply_lora_to_model(model, cfg).eval()


torch.manual_seed(0)
a = build()
a_sd = a.state_dict()
lora_sd = {k: v.clone() for k, v in a_sd.items() if "lora" in k}
base_sd = {k: v for k, v in a_sd.items() if "lora" not in k}
print("lora tensors:", len(lora_sd), "example:", next(iter(lora_sd)))
x = torch.zeros(1, 3, 1008, 1008)
with torch.no_grad():
    ref = a.backbone.forward_image(x)["vision_features"].clone()

for prefix, label in (("", "原样"), ("module.", "加 module."), ("backbone.", "加 backbone.")):
    payload = {"state_dict": {prefix + k: v for k, v in lora_sd.items()}}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        path = fh.name
    torch.save(payload, path)

    b = build()
    b.load_state_dict(base_sd, strict=False)     # 基础权重与 A 完全一致（LoRA 键另行加载）
    with torch.no_grad():                        # LoRA 置零，确保加载确实起作用
        for k, v in b.state_dict().items():
            if "lora" in k:
                v.zero_()
    matched = load_lora_weights(b, path)
    with torch.no_grad():
        out = b.backbone.forward_image(x)["vision_features"]
    ok = torch.allclose(ref, out, atol=1e-5)
    print(f"{label}: matched={matched} allclose={ok} maxdiff={float((ref - out).abs().max()):.3e}", flush=True)
    Path(path).unlink()
    del b
print("lora loader test done")
