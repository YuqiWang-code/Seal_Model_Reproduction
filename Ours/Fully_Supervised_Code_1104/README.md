# Fully_Supervised_Code_1104：全监督 RGB-T（Seal train 全像素 GT）

由原始代码改成全监督的版本：SAM3 与两套视觉模态 LoRA 冻结，训练融合模块 `M1`
与共享 LoRA3（`--no-lora3` 时只训 M1）。RGB/T 视觉特征经 M1 融合后，配合数据集
真实框做一次 Decoder，预测 mask 用 GT 做结构损失 + Sigmoid Focal + Dice。
模型选择用 Seal val 的滑窗像素 IoU@0.5，test 不参与。

## 从原目录补齐的东西

| 文件 | 说明 |
|---|---|
| `SAM3_LoRA/lora_layers.py` | 原仓库缺这个模块，按 `sam3/adapter.py` 的 `LoRALayer`/`DualLoRALinear` 数学重写：模态 LoRA（rank/alpha、kaiming down + zero up）+ 可挂共享 LoRA3 的 `LoRALinear`，并提供 `apply_lora_to_model`、`load_lora_weights` |
| `SAM3_LoRA/sam3/` | `Original_Code/sam3` 的副本（同一份 SAM3 源码，供 `sys.path` 使用） |
| `test_seal.py` | 原目录没有推理脚本；这里补上，产出与 GT 同名的 1,983 张 8 位 PNG |
| `data.py` | `load_t_image` 原式 `x - min/(max-min)*255` 在 16 位 TIFF 上会在转 uint8 时回绕，改为逐图 min-max |
| `train.py` | 设备由 `SEAL_CUDA` 控制；数据/权重默认路径改到本服务器 |

## 服务器运行

```bash
ROOT=/share_datasets/Seal_Model_Reproduction
PY=$ROOT/Ours/venv_ours/bin/python
cd $ROOT/Ours/Fully_Supervised_Code_1104

SEAL_CUDA=1 $PY train.py --epoch 30 --lr 5e-5 --img_size 1008 \
  --checkpoint $ROOT/Ours/weights/sam3.pt \
  --rgb_lora $ROOT/Ours/weights/RGB+LoRA/lora_weights.pt \
  --t_lora  $ROOT/Ours/weights/T+LoRA/lora_weights.pt \
  --train_data_dir $ROOT/Seal_data/train --val_data_dir $ROOT/Seal_data/val \
  --save_path train_logs 2>&1 | tee train_fullsup.log

SEAL_CUDA=1 $PY test_seal.py --weights train_logs/best.pth \
  --test_data_dir $ROOT/Seal_data/test \
  --out_dir $ROOT/results/Ours/FullySup/test_maps

cd $ROOT
$PY evaluate.py --pred_root $ROOT/results/Ours/FullySup/test_maps \
  --gt_dir $ROOT/Seal_data/test/GT --json_output $ROOT/results/Ours/FullySup/metrics4.json
$PY evaluate_small_targets.py --pred_root $ROOT/results/Ours/FullySup/test_maps \
  --gt_dir $ROOT/Seal_data/test/GT --json_output $ROOT/results/Ours/FullySup/metrics_small.json
```

## 需要放在 `Ours/weights/` 的文件

- `sam3.pt`（SAM3 主干/ tracker 权重，两个方法都要）
- `resnet50-19c8e357.pth`（仅原始版 T 分支；已放入）
- `RGB+LoRA/lora_weights.pt`、`T+LoRA/lora_weights.pt`（仅全监督版，冻结的模态 LoRA）
