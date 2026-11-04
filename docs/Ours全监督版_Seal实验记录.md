# Ours（全监督版，RGB-T 全像素 GT）：Seal 训练与测试记录

代码：`Ours/Fully_Supervised_Code_1104/`。由原始代码改成全监督：冻结 SAM3 与两套视觉
模态 LoRA，训练融合模块 `M1` 与共享 LoRA3；SAM3 的 Decoder 用数据集的真实框作几何提示，
预测 mask 用 GT 做结构损失 + Sigmoid Focal + Dice。训练、验证、测试全部使用 `Seal_data`。

## 协议

| 项目 | 值 |
|---|---|
| 输入 | Seal `train/RGB` + `train/T`（逐图 min-max → 8 位） |
| 监督 | `train/GT` 全像素 mask；`train/box` 仅作 Decoder 的几何提示 |
| 可训练参数 | `M1`（跨模态融合）+ 共享 LoRA3（视觉 qkv/proj，64 个 site）；SAM3 与模态 LoRA 冻结 |
| 损失 | 结构损失（加权 BCE + 加权 IoU）+ Sigmoid Focal（α=0.25, γ=2）+ Dice，等权 |
| 优化器 | AdamW，lr 5e-5，weight decay 0.01，30 epoch（每 50 epoch ×0.1 的日程） |
| 模型选择 | Seal **val** 500 张滑窗像素 IoU@0.5 最优；test 不参与 |
| 测试 | 同一入口（真实框提示）→ sigmoid → 双线性恢复 GT 尺寸 → 逐图 min-max → 8 位 PNG |
| 指标 | 根目录 `evaluate.py` + `evaluate_small_targets.py`，同一批 1,983 张 |

## 从原目录补齐/修正的内容

| 文件 | 说明 |
|---|---|
| `SAM3_LoRA/lora_layers.py` | 原仓库缺该模块；按 `sam3/adapter.py` 的 LoRA 数学重写：模态 LoRA（kaiming down + zero up，α/rank 缩放）+ 共享 LoRA3 挂载点，并提供 `apply_lora_to_model`、`load_lora_weights`（精确键 → 后缀匹配） |
| `SAM3_LoRA/sam3/` | 补上被 import 的 SAM3 源码（与 `Original_Code/sam3` 同一份） |
| `test_seal.py` | 原目录无推理脚本，新增；产出与 GT 同名 PNG |
| `data.py` | `load_t_image` 原式在 16 位 TIFF 上会 uint8 回绕，改为逐图 min-max |
| `train.py` | 设备由 `SEAL_CUDA` 控制；数据/权重默认路径指向本项目 |

## 环境与命令

```bash
bash /share_datasets/Seal_Model_Reproduction/Ours/run_fullsup.sh 30 1
```

权重：`Ours/weights/sam3.pt`、`Ours/weights/RGB+LoRA/lora_weights.pt`、
`Ours/weights/T+LoRA/lora_weights.pt`。

## 冒烟与基准

- LoRA 注入 64 个 site（32 block × qkv/proj），与代码中 `LORA3_LAYER_COUNT=64` 一致；
  共享 LoRA3 1.57M 参数，`M1` 1.19M；1008² 前向 0.24 s。
- 训练步（随机初始化）：约 0.45 s/step，峰值显存 9.8 GiB；1,500 步 ≈ 11 min/epoch，
  加 val 约 14 min/epoch。随机初始化时文本编码器输出 NaN，属无权重初始化的预期现象，
  加载 `sam3.pt` 后需复核首步 loss 有限。

## 结果

<!--RESULTS-->

## 产物

- 权重：`results/Ours/FullySup/best.pth`（`m1` + `lora3`）；日志 `logs/ours_fullsup_train.log`
- 预测：`results/Ours/FullySup/test_maps/`（1,983 张）
- 指标：`results/Ours/FullySup/metrics4.json`、`metrics_small.json`
