"""核实 SAM3 是否只能吃 1008×1008：用 512/1008 各跑一次 forward_image。"""
from pathlib import Path

import torch


def main():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("cm_lora", Path("CM-LoRA.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cm_lora"] = mod
    spec.loader.exec_module(mod)

    ckpt = Path("/share_datasets/Seal_Model_Reproduction/Ours/weights/sam3.pt")
    dev = torch.device("cuda")
    model = mod.build_frozen_base_model(ckpt, dev)
    for size in (1008, 512):
        x = torch.randn(1, 3, size, size, device=dev)
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.backbone.forward_image(x)
            fpn = out["backbone_fpn"][-1]
            print(f"{size}: OK, fpn[-1] {tuple(fpn.shape)}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc().strip().splitlines()
            print(f"{size}: FAIL {type(exc).__name__!r} msg={str(exc)[:120]!r}")
            for line in tb[-4:]:
                print("   ", line.strip())


if __name__ == "__main__":
    main()
