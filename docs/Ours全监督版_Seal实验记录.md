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

## 模态 LoRA 由本项目重训（原权重缺失）

原始工程里冻结的 `outputs/RGB+LoRA/best/lora_weights.pt` 与 `outputs/T+LoRA/best/lora_weights.pt`
未随代码提供，作者侧也已确认没有留存。因此这两套**模态 LoRA 由本项目在 Seal 上重训**，
全监督版仍按“冻结模态 LoRA + 只训 M1 与 LoRA3”的原设计运行。重训脚本
`Fully_Supervised_Code_1104/train_lora.py`，协议如下：

| 项目 | 值 |
|---|---|
| 结构 | SAM3 图像模型 + 视觉 `qkv`/`proj` 上的 LoRA（rank 16、alpha 32、64 site） |
| 可训参数 | 仅 LoRA（约 0.6M 张量 / 12.6 MB），SAM3 其余参数全部冻结 |
| 数据 | 单模态 Seal train（RGB 或 T；T 逐图 min-max → 8 位）1,500 张 |
| 监督 | Seal `train/box` 作 Decoder 几何提示 + Seal `train/GT` 全像素 mask 的结构损失 + Focal + Dice |
| 优化器 | AdamW，lr 2e-4，8 epoch，batch 1 |
| 选点 | 每 2 epoch 在 Seal val 500 张算像素 IoU@0.5，取最优；test 不参与 |
| 产出 | `Ours/weights/{RGB+LoRA,T+LoRA}/lora_weights.pt`（`{"state_dict": ...}`，唯一后缀可加载） |

冒烟（2 张训练样本、1 epoch）即达到 val IoU 0.775，说明 SAM3 预训练特征与框提示链路正常。
这是对原方法的**替代实现**，须在论文与表备注中写明“模态 LoRA 由本项目在 Seal train 重训”。

## 环境与命令

```bash
bash /share_datasets/Seal_Model_Reproduction/Ours/run_fullsup.sh 30 1
```

权重（服务器 `/share_datasets/Seal_Model_Reproduction/Ours/weights/`，3.21 GiB 主干不进本仓库，本地副本已删除）：
`sam3.pt`（3,450,062,241 字节，SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`，服务器与本地曾逐字节一致）、
`RGB+LoRA/lora_weights.pt`、`T+LoRA/lora_weights.pt`（后两者由 `train_lora.py` 在 Seal 上重训）。

## 冒烟与基准

- LoRA 注入 64 个 site（32 block × qkv/proj），与代码中 `LORA3_LAYER_COUNT=64` 一致；
  共享 LoRA3 1.57M 参数，`M1` 1.19M；1008² 前向 0.24 s。
- 训练步（随机初始化）：约 0.45 s/step，峰值显存 9.8 GiB；1,500 步 ≈ 11 min/epoch，
  加 val 约 14 min/epoch。随机初始化时文本编码器输出 NaN，属无权重初始化的预期现象，
  加载 `sam3.pt` 后需复核首步 loss 有限。

## 结果

30 epoch、lr 5e-5、batch 1、1008×1008，17:06 结束。val 像素 IoU 在 **第 20 epoch 达到最高 0.51724**
（对应 val MAE 0.000122），之后在 0.5000–0.5102 波动，最终权重即 ep20 的 `best.pth`（11 MB：`m1`+`lora3`）。

| 来源 | S↑ | Fw↑ | E↑ | MAE↓ | IoU↑ | F1↑ | Pd↑ | Fa↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全监督版，Seal train 全像素 GT，val 最优（ep20），test 1,983 张 | 0.49924 | 0.54263 | 0.68021 | 0.00025726 | 0.43005 | 0.60145 | 0.86320 | 5.0377e-05 |

完整精度：S 0.4992378406777436、Fw 0.5426273297198246、E 0.6802070573397926、
MAE 0.0002572631055954844、IoU 0.43004989835520235、F1 0.6014474024295684、
Pd 0.863199241500745、Fa 5.03773154185837e-05；像素 TP/FP/FN = 81,445 / 83,673 / 24,267
（TP+FN = 105,712 = GT 前景像素）；GT 组件 7,383、预测组件 8,173、匹配成功 6,373、未匹配面积 25,911 px。

与原始版（同协议下框监督、val MAE 选点）相比，全监督版在六项上大幅领先：IoU 0.24975→**0.43005**、
F1 0.39968→**0.60145**、Pd 0.58838→**0.86320**、Fw 0.38774→**0.54263**、E 0.57970→**0.68021**、
MAE 0.000279→**0.000257**；Fa 9.78e-05→**5.04e-05**（减半）。

与 12 个全监督 IRSTD 基线相比：IoU 0.43005 高于全部基线（最高 BGM 0.30735），
F1 0.60145 高于全部（最高 BGM 0.47019），Pd 0.86320 高于全部（最高 HDNet 0.82135），
Fw 0.54263 略高于最高的 MSHNet 0.54104，E 0.68021 处于基线区间中部（0.66410–0.68601，
低于 MSHNet/ILNet/UIU-Net/DNA-Net），Fa 5.04e-05 处于基线区间中部（1.93e-05–1.04e-04）。

说明：本行使用本项目重训的两套模态 LoRA（见上节），且推理时与训练一致地用 Seal test 的
真实框作为 Decoder 几何提示（`Seal_data/test/box`，Seal 官方标注），不读取任何 GT mask。

## 产物

- 权重：`results/Ours/FullySup/best.pth`（`m1` + `lora3`）；日志 `logs/ours_fullsup_train.log`
- 预测：`results/Ours/FullySup/test_maps/`（1,983 张）
- 指标：`results/Ours/FullySup/metrics4.json`、`metrics_small.json`
