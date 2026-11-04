#!/bin/bash
# 集成测试：迷你 Seal 子集（24/6/6）把「训练→选点→推理→八项指标」整条链路跑通。
# 全部用随机初始化/哑权重，数值无意义，只验证管线。
set -e
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
SEAL=$ROOT/Seal_data
MINI=/tmp/seal_mini
export SEAL_CUDA=1

echo "=== 1. 迷你数据集 ==="
rm -rf "$MINI"
for spec in train:24 val:6 test:6; do
  split=${spec%%:*}; n=${spec##*:}
  for d in RGB T box GT; do mkdir -p "$MINI/$split/$d"; done
  ls "$SEAL/$split/GT" | sed 's/\.png$//' | head -n "$n" > /tmp/stems_$split.txt
  while read -r s; do
    ln -sf "$SEAL/$split/RGB/$s.jpg" "$MINI/$split/RGB/$s.jpg"
    ln -sf "$SEAL/$split/T/$s.tif"   "$MINI/$split/T/$s.tif"
    ln -sf "$SEAL/$split/box/$s.txt" "$MINI/$split/box/$s.txt"
    ln -sf "$SEAL/$split/GT/$s.png"  "$MINI/$split/GT/$s.png"
  done < /tmp/stems_$split.txt
  echo "$split: $(ls "$MINI/$split/GT" | wc -l) 张"
done

echo "=== 2. 哑 SAM3 权重（只让加载函数通过，实际仍是随机初始化） ==="
$PY - <<'PY'
import torch
torch.save({"detector.dummy": torch.zeros(1), "tracker.dummy": torch.zeros(1)}, "/tmp/dummy_sam3.pt")
print("wrote /tmp/dummy_sam3.pt")
PY

echo "=== 3. 原始版：train.py 1 epoch ==="
cd "$ROOT/Ours/Original_Code"
$PY train.py --epoch 1 --lr 5e-5 --img_size 1008 --batchsize 1 \
  --checkpoint /tmp/dummy_sam3.pt --resnet_ckpt "$ROOT/Ours/weights/resnet50-19c8e357.pth" \
  --train_data_dir "$MINI/train" --val_data_dir "$MINI/val" \
  --save_path /tmp/ours_mini_orig --save_name best3.pth 2>&1 | tail -3

echo "=== 4. 原始版：test.py ==="
$PY test.py --mode single --checkpoint /tmp/ours_mini_orig/best3.pth \
  --test_data_dir "$MINI/test" --resnet_ckpt "$ROOT/Ours/weights/resnet50-19c8e357.pth" \
  --save_dir /tmp/ours_mini_orig/infer 2>&1 | tail -3
MAPS=/tmp/ours_mini_orig/infer/best3/test
echo "预测张数: $(ls "$MAPS"/*.png | wc -l)"

echo "=== 5. 原始版：八项指标 ==="
cd "$ROOT"
$PY evaluate.py --pred_root "$MAPS" --gt_dir "$MINI/test/GT" --json_output /tmp/ours_mini_orig/m4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root "$MAPS" --gt_dir "$MINI/test/GT" --json_output /tmp/ours_mini_orig/small.json 2>&1 | grep -A6 '"metrics"'

echo "=== 6. 全监督版：构造哑模态 LoRA ==="
cd "$ROOT/Ours/Fully_Supervised_Code_1104"
$PY - <<'PY'
import sys
from pathlib import Path
import torch

ROOT = Path("/share_datasets/Seal_Model_Reproduction")
HERE = ROOT / "Ours" / "Fully_Supervised_Code_1104"
sys.path.insert(0, str(HERE / "SAM3_LoRA"))
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model

bpe = HERE / "SAM3_LoRA" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.0, target_modules=["qkv", "proj"])
torch.manual_seed(0)
m = apply_lora_to_model(
    build_sam3_image_model(bpe_path=str(bpe), device="cpu", eval_mode=True, checkpoint_path=None,
                           load_from_HF=False, enable_segmentation=True,
                           enable_inst_interactivity=False, compile=False), cfg)
sd = {k: v.clone() for k, v in m.state_dict().items() if "lora" in k}
out = ROOT / "Ours" / "weights" / "_dummy_lora"
(out / "RGB+LoRA").mkdir(parents=True, exist_ok=True)
(out / "T+LoRA").mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": sd}, out / "RGB+LoRA" / "lora_weights.pt")
torch.save({"state_dict": {k: v * 2 for k, v in sd.items()}}, out / "T+LoRA" / "lora_weights.pt")
print("dummy lora written:", len(sd), "tensors")
PY

echo "=== 7. 全监督版：train.py 1 epoch ==="
$PY train.py --epoch 1 --lr 5e-5 --img_size 1008 \
  --checkpoint /tmp/dummy_sam3.pt \
  --rgb_lora "$ROOT/Ours/weights/_dummy_lora/RGB+LoRA/lora_weights.pt" \
  --t_lora  "$ROOT/Ours/weights/_dummy_lora/T+LoRA/lora_weights.pt" \
  --train_data_dir "$MINI/train" --val_data_dir "$MINI/val" \
  --save_path /tmp/ours_mini_fullsup 2>&1 | tail -4

echo "=== 8. 全监督版：test_seal.py ==="
$PY test_seal.py --weights /tmp/ours_mini_fullsup/best.pth \
  --checkpoint /tmp/dummy_sam3.pt \
  --rgb_lora "$ROOT/Ours/weights/_dummy_lora/RGB+LoRA/lora_weights.pt" \
  --t_lora  "$ROOT/Ours/weights/_dummy_lora/T+LoRA/lora_weights.pt" \
  --test_data_dir "$MINI/test" --out_dir /tmp/ours_mini_fullsup/test_maps --num_workers 0 2>&1 | tail -3
MAPS=/tmp/ours_mini_fullsup/test_maps
echo "预测张数: $(ls "$MAPS"/*.png | wc -l)"

echo "=== 9. 全监督版：八项指标 ==="
cd "$ROOT"
$PY evaluate.py --pred_root "$MAPS" --gt_dir "$MINI/test/GT" --json_output /tmp/ours_mini_fullsup/m4.json 2>&1 | tail -8
$PY evaluate_small_targets.py --pred_root "$MAPS" --gt_dir "$MINI/test/GT" --json_output /tmp/ours_mini_fullsup/small.json 2>&1 | grep -A6 '"metrics"'

echo "=== 集成测试完成 ==="
