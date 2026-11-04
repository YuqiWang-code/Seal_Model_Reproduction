from pathlib import Path
p = Path("tmp/test_lora_loader.py")
s = p.read_text(encoding="utf-8")
s = s.replace("b.load_state_dict(base_sd, strict=True)      # 基础权重与 A 完全一致",
              "b.load_state_dict(base_sd, strict=False)     # 基础权重与 A 完全一致（LoRA 键另行加载）")
p.write_text(s, encoding="utf-8")
print("patched")
